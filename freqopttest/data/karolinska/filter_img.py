import os 
import shutil

def copy_faces(base_dir, dest_base, img_fname_filter, flatten=True):
    """
    If flatten=True, 
    Copy files. Remove dirs of subjects so that files of all subjects are 
    in the same folder.
    """
    for fname in os.listdir(base_dir):
        subject_dir = os.path.join(base_dir, fname)
        if os.path.isdir(subject_dir):
            for img_fname in os.listdir(subject_dir):
                if img_fname.lower().endswith('.jpg'):
                    # copy
                    img_src = os.path.join(subject_dir, img_fname)
                    if flatten:
                        dest_dir = dest_base
                    else:
                        dest_dir = os.path.join(dest_base, subject_dir)
                    img_dest = os.path.join(dest_dir, img_fname)
                    #print img_fname
                    if img_fname_filter(img_fname):
                        try:
                            os.makedirs(dest_dir)
                        except OSError:
                            pass
                        # image name passes the filter
                        # Then, copy it
                        print('copying: %s -> %s'%(img_src, img_dest))
                        shutil.copy(img_src, img_dest)


def copy_by_expression(base_flat_dir, dest_base):
    """Copy faces into another folder. Group by expression types.
    - base_flat_dir: folder containing all images without sub-directories
    """
    for fname in os.listdir(base_flat_dir):
        if fname.endswith('.JPG'):
            # copy
            img_src = os.path.join(base_flat_dir, fname)
            img_name = fname[:fname.index('.JPG')]
            cat = img_name[4:6]
            dest_dir = os.path.join(dest_base, cat)
            img_dest = os.path.join(dest_dir, fname)
            try:
                os.makedirs(dest_dir)
            except OSError:
                pass
            # image name passes the filter
            # Then, copy it
            print('copying: %s -> %s'%(img_src, img_dest))
            shutil.copy(img_src, img_dest)


def straight_face_filter(fname):
    # remove .jpg
    ind = fname.lower().index('.jpg')
    name = fname[:ind]
    return name.endswith('S')


def copy_flat():
    base_dir = 'KDEF'
    dest_base = 'KDEF_straight'
    img_fname_filter = straight_face_filter
    copy_faces(base_dir, dest_base, img_fname_filter, True)

def copy_group():
    size = 48
    #base_flat = 'crop_%d'%size
    #dest_base = 'crop_%d_group'%size
    base_flat = 'S_crop'
    dest_base = 'S_crop_group'
    copy_by_expression(base_flat, dest_base)


if __name__ == '__main__':
    copy_group()





