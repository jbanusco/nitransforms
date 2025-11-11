import numpy as np
import copy
import itk


def compute_jacobian(displacement_field: np.array, 
                     spacing: np.array = np.array([1, 1, 1])):
    """ Compute the Jacobian of the displacement field """

    jacobian = np.zeros(displacement_field.shape[:-1] + (3, 3))
    dx, dy, dz = np.gradient(displacement_field, spacing[0], spacing[1], spacing[2], axis=(0, 1, 2), edge_order=1)

    jacobian[..., 0, 0] = 1 + dx[..., 0]
    jacobian[..., 0, 1] = dx[..., 1]
    jacobian[..., 0, 2] = dx[..., 2]

    jacobian[..., 1, 0] = dy[..., 0]
    jacobian[..., 1, 1] = 1 + dy[..., 1]
    jacobian[..., 1, 2] = dy[..., 2]

    jacobian[..., 2, 0] = dz[..., 0]
    jacobian[..., 2, 1] = dz[..., 1]
    jacobian[..., 2, 2] = 1 + dz[..., 2]

    jacobian_determinant = np.linalg.det(jacobian)
    return jacobian_determinant


def compute_cc(fixed_img: itk.image, 
               moving_img: itk.image, 
               radius: int = 4, 
               img_type: itk.itkCType = itk.F, 
               dim: int = 3, 
               mask: itk.image = None):
    """ Compute the cross-correlation between the two images """

    ImageType = itk.Image[img_type, dim]
    CorrelationMetricType = itk.ANTSNeighborhoodCorrelationImageToImageMetricv4[ImageType, ImageType].New()
    CorrelationMetricType.SetRadius(radius)

    CorrelationMetricType.SetMovingImage(moving_img)
    CorrelationMetricType.SetUseMovingImageGradientFilter(False)
    if mask is not None:
        movingImageMask = copy.deepcopy(mask)
        CorrelationMetricType.SetMovingImageMask(movingImageMask)

    CorrelationMetricType.SetFixedImage(fixed_img)    
    CorrelationMetricType.SetUseFixedImageGradientFilter(False)
    if mask is not None:
        fixedImageMask = copy.deepcopy(mask)
        CorrelationMetricType.SetFixedImageMask(fixedImageMask)

    CorrelationMetricType.SetVirtualDomainFromImage(fixed_img)
    CorrelationMetricType.Initialize()

    return CorrelationMetricType.GetValue()


def compute_mi(fixed_img: itk.image, 
               moving_img: itk.image, 
               num_bins: int = 32, 
               img_type: itk.itkCType = itk.F, 
               dim: int = 3, 
               mask: itk.image = None):
    """ Compute the mutual information between the two images """

    ImageType = itk.Image[img_type, dim]

    MutualInformation = itk.MattesMutualInformationImageToImageMetricv4[ImageType, ImageType].New()
    MutualInformation.SetNumberOfHistogramBins(num_bins)

    MutualInformation.SetMovingImage(moving_img)    
    MutualInformation.SetUseMovingImageGradientFilter(False)
    if mask is not None:
        movingImageMask = copy.deepcopy(mask)
        MutualInformation.SetMovingImageMask(movingImageMask)

    MutualInformation.SetFixedImage(fixed_img)    
    MutualInformation.SetUseFixedImageGradientFilter(False)
    if mask is not None:
        fixedImageMask = copy.deepcopy(mask)
        MutualInformation.SetFixedImageMask(fixedImageMask)

    MutualInformation.SetVirtualDomainFromImage(fixed_img)
    MutualInformation.Initialize()

    return MutualInformation.GetValue()