"""
This script is used to generate slicewise training data
@author:Karsten Roth - Heidelberg University, 07/11/2017
"""


"""=============================================="""
"""========== COPMUTE WEIGHTMAP ================="""
"""=============================================="""
def find_borders(liver_mask, lesion_mask, save_path_liver, save_path_lesion, width=5):
    struct_elem     = np.ones([width,width])

    ### 勾勒肝脏边缘
    ndi.binary_dilation(liver_mask, struct_elem)
    outer_border = ndi.binary_dilation(liver_mask, struct_elem).astype(int)-liver_mask
    inner_border = liver_mask-ndi.binary_erosion(liver_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### 生成一个权重图，将权重放在靠近边界的像素上
    weight = 1/np.sqrt(3)
    boundary_weightmap_liver = 0.75*np.clip(np.exp(-0.02*(ndi.morphology.distance_transform_edt(1-total_border))),weight,None)+0.25*liver_mask


    ### Locate pixels around lesion boundaries
    ndi.binary_dilation(lesion_mask, struct_elem)
    outer_border = ndi.binary_dilation(lesion_mask, struct_elem).astype(int)-lesion_mask
    inner_border = lesion_mask-ndi.binary_erosion(lesion_mask, struct_elem).astype(int)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight = 1/np.sqrt(3)
    boundary_weightmap_lesion = 0.75*np.clip(np.exp(-0.02*(ndi.morphology.distance_transform_edt(1-total_border))),weight,None)+0.25*lesion_mask
    boundary_weightmap_liver  = 0.65*boundary_weightmap_liver + 0.35*boundary_weightmap_lesion


    np.save(save_path_liver, boundary_weightmap_liver.astype(np.float16))
    np.save(save_path_lesion,boundary_weightmap_lesion.astype(np.float16))







"""=============================================="""
"""========== MAIN GENERATION FILE =============="""
"""=============================================="""
def main(opt):
    if not os.path.exists(opt.save_path_4_training_slices): os.makedirs(opt.save_path_4_training_slices)
    # 为切好的训练测试集在各自的路径（../LOADDATA/Training_Data_2D or../LOADDATA/Test_Data_2D）里创建 Assign_2D_Volumes.csv
    # 以保存 Train & Test volume 的切片以及对应路径
    assign_file_v    = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_Volumes.csv",    ["Volume","Slice Path"])

    """
    并且为2D训练数据创建 Assign_2D_LiverMasks.csv
                        Assign_2D_LesionMasks.csv
    以保存volume_.nii切片的路径以及此切片是否有肝脏/肿瘤
                    创建 Assign_2D_LiverWmaps.csv
                        Assign_2D_LesionWmaps.csv
    以保存volume_.nii每个切片肝脏、肿瘤边缘权重数据的相应路径
    """
    if not opt.is_test_data:
        assign_file_les  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionMasks.csv",["Volume","Slice Path","Has Mask"])
        assign_file_liv  = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverMasks.csv", ["Volume","Slice Path","Has Mask"])
        assign_file_wliv = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LiverWmaps.csv", ["Volume","Slice Path"])
        assign_file_wles = gu.CSVlogger(opt.save_path_4_training_slices+"/Assign_2D_LesionWmaps.csv",["Volume","Slice Path"])

    volumes  = os.listdir(opt.path_2_training_volumes)
    segs, vols = [],[]

    for x in volumes:
        if 'segmentation' in x and not opt.is_test_data: segs.append(x)
        if 'volume' in x: vols.append(x)

    vols.sort()
    segs.sort()


    if not os.path.exists(opt.save_path_4_training_slices):
        os.makedirs(opt.save_path_4_training_slices)


    if not opt.is_test_data:
        volume_iterator = tqdm(zip(vols, segs), position=0, total=len(vols))
    else:
        volume_iterator = tqdm(vols, position=0)


    if opt.is_test_data:
        volume_info = {}

    for i,data_tuple in enumerate(volume_iterator):
        ### ASSIGNING RELEVANT VARIABLES
        if not opt.is_test_data:
            vol, seg = data_tuple
        else:
            vol = data_tuple

        ### LOAD VOLUME AND MASK DATA
        volume_iterator.set_description('Loading Data...')

        volume      = nib.load(opt.path_2_training_volumes+"/"+vol)
        v_name      = vol.split(".")[0]

        if opt.is_test_data:
            header, affine = volume.header, volume.affine
            volume_info[v_name] = {'header':header, 'affine':affine}

        volume         = np.array(volume.dataobj)   # nibabel 读取数据的顺序是 (h, w, num_silce)
        volume         = volume.transpose(2,0,1)    # (num_slice, h, w)
        save_path_v    = opt.save_path_4_training_slices+"/Volumes/"+v_name
        if not os.path.exists(save_path_v): os.makedirs(save_path_v)

        if not opt.is_test_data:
            segmentation  = np.array(nib.load(opt.path_2_training_volumes+"/"+seg).dataobj)
            segmentation  = segmentation.transpose(2,0,1)

            save_path_lesion_masks        = opt.save_path_4_training_slices+"/LesionMasks/"+v_name
            save_path_liver_masks         = opt.save_path_4_training_slices+"/LiverMasks/"+v_name
            save_path_lesion_weightmaps   = opt.save_path_4_training_slices+"/BoundaryMasksLesion/"+v_name
            save_path_liver_weightmaps    = opt.save_path_4_training_slices+"/BoundaryMasksLiver/"+v_name

            if not os.path.exists(save_path_lesion_masks):     os.makedirs(save_path_lesion_masks)
            if not os.path.exists(save_path_liver_masks):      os.makedirs(save_path_liver_masks)
            if not os.path.exists(save_path_lesion_weightmaps):os.makedirs(save_path_lesion_weightmaps)
            if not os.path.exists(save_path_liver_weightmaps): os.makedirs(save_path_liver_weightmaps)


        volume_iterator.set_description('Generating Weightmaps and saving slices...')

        if not opt.is_test_data:
            volume_slice_iterator = tqdm(zip(volume, segmentation), position=1, total=len(volume))
        else:
            volume_slice_iterator = tqdm(volume, position=1)


        for idx,data_tuple in enumerate(volume_slice_iterator):

            if not opt.is_test_data:
                (v_slice, s_slice) = data_tuple
                liver_mask  = s_slice>=1
                lesion_mask = s_slice==2

                find_borders(liver_mask, lesion_mask, save_path_liver_weightmaps +"/slice-"+str(idx)+".npy", save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy")
                assign_file_wles.write([v_name, save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy"])
                assign_file_wliv.write([v_name, save_path_liver_weightmaps +"/slice-"+str(idx)+".npy"])

                np.save(save_path_lesion_masks+"/slice-"+str(idx)+".npy", lesion_mask.astype(np.uint8))
                np.save(save_path_liver_masks +"/slice-"+str(idx)+".npy", liver_mask.astype(np.uint8))
                assign_file_les.write([v_name, save_path_lesion_masks+"/slice-"+str(idx)+".npy", 1 in lesion_mask.astype(np.uint8)])
                assign_file_liv.write([v_name, save_path_liver_masks +"/slice-"+str(idx)+".npy", 1 in liver_mask.astype(np.uint8)])
            else:
                v_slice = data_tuple

            np.save(save_path_v+"/slice-"+str(idx)+".npy", v_slice.astype(np.int16))
            assign_file_v.write([v_name, save_path_v+"/slice-"+str(idx)+".npy"])

        if opt.is_test_data: pkl.dump(volume_info, open(opt.save_path_4_training_slices+'/volume_nii_info.pkl','wb'))










"""=========================================="""
"""========== ______MAIN______ =============="""
"""=========================================="""
if __name__ == '__main__':

    """=================================="""
    ### LOAD BASIC LIBRARIES
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np, os, sys, nibabel as nib, argparse, pickle as pkl
    os.chdir(os.path.dirname(os.path.realpath(__file__)))    
    sys.path.insert(0, os.getcwd()+'/../Utilities')
    import General_Utilities as gu
    from tqdm import tqdm
    import scipy.ndimage as ndi


    """===================================="""
    ### GET PATHS
    #Read network and training setup from text file.
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--path_2_training_volumes',     type=str, default='placeholder',
                          help='Path to original LiTS-volumes in nii-format.')
    parse_in.add_argument('--save_path_4_training_slices', type=str, default='placeholder',
                          help='Where to save the 2D-conversion.')
    parse_in.add_argument('--is_test_data', action='store_true',
                          help='Flag to mark if input data is test data or not.')
    opt = parse_in.parse_args()


    """===================================="""
    ### RUN GENERATION
    if 'placeholder' in opt.path_2_training_volumes:
        pt = 'Test_Data' if opt.is_test_data else 'Training_Data'
        # path_2_training_volumes = ../OriginalData/Training_Data or ../OriginalData/Test_Data
        # 原始数据集 .nii 路径
        opt.path_2_training_volumes     = os.getcwd()+'/../OriginalData/'+pt
    if 'placeholder' in opt.save_path_4_training_slices:
        pt = 'Test_Data_2D' if opt.is_test_data else 'Training_Data_2D'
        # save_path_4_training_slices = ../LOADDATA/Training_Data_2D or../LOADDATA/Test_Data_2D
        # 切成2D切片的数据集存放路径
        opt.save_path_4_training_slices = os.getcwd()+'/../LOADDATA/'+pt


    """===================================="""
    ### RUN GENERATION
    main(opt)
