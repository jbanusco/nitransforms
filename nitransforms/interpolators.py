
import numpy as np
import nibabel as nib

from nitransforms.base import (   
    TransformBase,     
    ImageGrid,    
    _as_homogeneous,
)

from scipy.ndimage import map_coordinates

def genericlabel_interpolator(labelImage: nib.Nifti1Image, 
                              xfm: TransformBase, 
                              order: int = 1, 
                              mode: str = 'constant'):
    """ Resample a multi-labeled image using a generic transform. """

    data = np.asanyarray(
        labelImage.dataobj,
        dtype=labelImage.get_data_dtype()
    )
    reference = xfm.reference
    targets = ImageGrid(labelImage).index(  # data should be an image
        _as_homogeneous(xfm.map(reference.ndcoords.T), dim=reference.ndim)
    )
    
    best_values = np.zeros_like(data)
    final_labels = np.zeros_like(data)    
    original_binary_labels = []
    list_binary_labels = []
    for label in np.unique(data.flatten()):
        # 1 interpolator per binary label
        binary_label = np.zeros_like(data)
        binary_label[data == label] = 1
        assert binary_label.max() == 1, "Error in binary label"
        original_binary_labels.append(binary_label)

        # Here you do the mapping
        resampled = map_coordinates(
            binary_label,
            targets.T,
            output=None,
            order=order,
            mode=mode,
            cval=0,
            prefilter=True,
        )

        # 2 keep label with highest value
        resampled = resampled.reshape(labelImage.shape)
        list_binary_labels.append(resampled)
        final_labels[resampled > best_values] = label
        best_values[resampled > best_values] = resampled[resampled > best_values]

    return nib.Nifti1Image(final_labels, affine=labelImage.affine, header=labelImage.header)