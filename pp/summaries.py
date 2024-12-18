import centrosome.cpmorphology
import centrosome.zernike
import numpy
import scipy.ndimage
import skimage.measure
import numpy as np
import pandas as pd
from enum import Enum
from tqdm import tqdm

"""

See the *Technical Notes* below for an explanation of a key step
underlying many of the following metrics: creating an
ellipse with the same second-moments as each object.

-  *Area:* *(2D only)* The number of pixels in the region.
-  *Volume:* *(3D only)* The number of voxels in the region.
-  *Perimeter:* *(2D only)* The total number of pixels around the boundary of each
   region in the image.
-  *SurfaceArea:* *(3D only)* The total number of voxels around the boundary of
   each region in the image.
-  *FormFactor:* *(2D only)* Calculated as 4\*π\*Area/Perimeter\ :sup:`2`. Equals 1
   for a perfectly circular object.
-  *Convex Area:* The area of a convex polygon containing the whole object.
   Best imagined as a rubber band stretched around the object. 
-  *Solidity:* The proportion of the pixels in the convex hull that are
   also in the object, i.e., *ObjectArea/ConvexHullArea*.
-  *Extent:* The proportion of the pixels (2D) or voxels (3D) in the bounding box
   that are also in the region. Computed as the area/volume of the object divided
   by the area/volume of the bounding box.
-  *EulerNumber:* The number of objects in the region minus the number
   of holes in those objects, assuming 8-connectivity.
-  *Center\_X, Center\_Y, Center\_Z:* The *x*-, *y*-, and (for 3D objects) *z-*
   coordinates of the point farthest away from any object edge (the *centroid*).
   Note that this is not the same as the *Location-X* and *-Y* measurements
   produced by the **Identify** or **Watershed**
   modules or the *Location-Z* measurement produced by the **Watershed** module.
-  *BoundingBoxMinimum/Maximum\_X/Y/Z:* The minimum/maximum *x*-, *y*-, and (for 3D objects)
   *z-* coordinates of the object.
-  *BoundingBoxArea:* *(2D only)* The area of a box containing the object.
-  *BoundingBoxVolume:* *(3D only)* The volume of a box containing the object.
-  *Eccentricity:* *(2D only)* The eccentricity of the ellipse that has the same
   second-moments as the region. The eccentricity is the ratio of the
   distance between the foci of the ellipse and its major axis length.
   The value is between 0 and 1. (0 and 1 are degenerate cases; an
   ellipse whose eccentricity is 0 is actually a circle, while an
   ellipse whose eccentricity is 1 is a line segment.)

    |MOSS_image0|


-  *MajorAxisLength:* The length (in pixels) of the major axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *MinorAxisLength:* The length (in pixels) of the minor axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *EquivalentDiameter:* The diameter of a circle or sphere with the same area
   as the object.
-  *Orientation:* *(2D only)* The angle (in degrees ranging from -90 to 90 degrees)
   between the x-axis and the major axis of the ellipse that has the
   same second-moments as the region.
-  *Compactness:* *(2D only)* Calculated as Perimeter\ :sup:`2`/4\*π\*Area, related to 
   Form Factor. A filled circle will have a compactness of 1, with irregular objects or 
   objects with holes having a value greater than 1.
-  *MaximumRadius:* *(2D only)* The maximum distance of any pixel in the object to
   the closest pixel outside of the object. For skinny objects, this is
   1/2 of the maximum width of the object.
-  *MedianRadius:* *(2D only)* The median distance of any pixel in the object to the
   closest pixel outside of the object.
-  *MeanRadius:* *(2D only)* The mean distance of any pixel in the object to the
   closest pixel outside of the object.
-  *MinFeretDiameter, MaxFeretDiameter:* *(2D only)* The Feret diameter is the
   distance between two parallel lines tangent on either side of the
   object (imagine taking a caliper and measuring the object at various
   angles). The minimum and maximum Feret diameters are the smallest and
   largest possible diameters, rotating the calipers along all possible
   angles.
-  *Zernike shape features:* *(2D only)* These metrics of shape describe a binary object
   (or more precisely, a patch with background and an object in the
   center) in a basis of Zernike polynomials, using the coefficients as
   features (*Boland et al., 1998*). Currently, Zernike polynomials from
   order 0 to order 9 are calculated, giving in total 30 measurements.
   While there is no limit to the order which can be calculated (and
   indeed you could add more by adjusting the code), the higher order
   polynomials carry less information.
-  *Spatial Moment features:* *(2D only)* A series of weighted averages 
   representing the shape, size, rotation and location of the object.
-  *Central Moment features:* *(2D only)* Similar to spatial moments, but
   normalized to the object's centroid. These are therefore not influenced
   by an object's location within an image.
-  *Normalized Moment features:* *(2D only)* Similar to central moments,
   but further normalized to be scale invariant. These moments are therefore
   not impacted by an object's size (or location).
-  *Hu Moment features:* *(2D only)* Hu's set of image moment features. These
   are not altered by the object's location, size or rotation. This means that
   they primarily describe the shape of the object.
-  *Inertia Tensor features:* *(2D only)* A representation of rotational
   inertia of the object relative to it's center.
-  *Inertia Tensor Eigenvalues features:* *(2D only)* Values describing 
   the movement of the Inertia Tensor array.



Technical notes
^^^^^^^^^^^^^^^

A number of the object measurements are generated by creating an ellipse
with the same second-moments as the original object region. This is
essentially the best-fitting ellipse for a given object with the same
statistical properties. Furthermore, they are not affected by the
translation or uniform scaling of a region.

Following computer vision conventions, the origin of the X and Y axes is at the top
left of the image rather than the bottom left; the orientation of objects whose topmost point
is on their right (or are rotated counter-clockwise from the horizontal) will therefore
have a negative orientation, while objects whose topmost point is on their left
(or are rotated clockwise from the horizontal) will have a positive orientation.

The Zernike features are computed within the minimum enclosing circle of
the object, i.e., the circle of the smallest diameter that contains all
of the object’s pixels.

References
^^^^^^^^^^

-  Rocha L, Velho L, Carvalho PCP, “Image moments-based structuring and
   tracking of objects”, Proceedings from XV Brazilian Symposium on
   Computer Graphics and Image Processing, 2002. `(pdf)`_
-  Principles of Digital Image Processing: Core Algorithms
   (Undergraduate Topics in Computer Science): `Section 2.4.3 -
   Statistical shape properties`_
-  Chrystal P (1885), “On the problem to construct the minimum circle
   enclosing n given points in a plane”, *Proceedings of the Edinburgh
   Mathematical Society*, vol 3, p. 30
-  Hu MK (1962), “Visual pattern recognition by moment invariants”, *IRE
   transactions on information theory*, 8(2), pp.179-187 `(link)`_

.. _(pdf): http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
.. _Section 2.4.3 - Statistical shape properties: http://www.scribd.com/doc/58004056/Principles-of-Digital-Image-Processing#page=49
.. _(link): https://ieeexplore.ieee.org/abstract/document/1057692

"""

AREA_SHAPE = "AreaShape"

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
ZERNIKE_N = 9

F_AREA = "Area"
F_PERIMETER = "Perimeter"
F_VOLUME = "Volume"
F_SURFACE_AREA = "SurfaceArea"
F_ECCENTRICITY = "Eccentricity"
F_SOLIDITY = "Solidity"
F_CONVEX_AREA = "ConvexArea"
F_EXTENT = "Extent"
F_CENTER_X = "Center_X"
F_CENTER_Y = "Center_Y"
F_CENTER_Z = "Center_Z"
F_BBOX_AREA = "BoundingBoxArea"
F_BBOX_VOLUME = "BoundingBoxVolume"
F_MIN_X = "BoundingBoxMinimum_X"
F_MAX_X = "BoundingBoxMaximum_X"
F_MIN_Y = "BoundingBoxMinimum_Y"
F_MAX_Y = "BoundingBoxMaximum_Y"
F_MIN_Z = "BoundingBoxMinimum_Z"
F_MAX_Z = "BoundingBoxMaximum_Z"
F_EULER_NUMBER = "EulerNumber"
F_FORM_FACTOR = "FormFactor"
F_MAJOR_AXIS_LENGTH = "MajorAxisLength"
F_MINOR_AXIS_LENGTH = "MinorAxisLength"
F_ORIENTATION = "Orientation"
F_COMPACTNESS = "Compactness"
F_INERTIA = "InertiaTensor"
F_MAXIMUM_RADIUS = "MaximumRadius"
F_MEDIAN_RADIUS = "MedianRadius"
F_MEAN_RADIUS = "MeanRadius"
F_MIN_FERET_DIAMETER = "MinFeretDiameter"
F_MAX_FERET_DIAMETER = "MaxFeretDiameter"

F_CENTRAL_MOMENT_0_0 = "CentralMoment_0_0"
F_CENTRAL_MOMENT_0_1 = "CentralMoment_0_1"
F_CENTRAL_MOMENT_0_2 = "CentralMoment_0_2"
F_CENTRAL_MOMENT_0_3 = "CentralMoment_0_3"
F_CENTRAL_MOMENT_1_0 = "CentralMoment_1_0"
F_CENTRAL_MOMENT_1_1 = "CentralMoment_1_1"
F_CENTRAL_MOMENT_1_2 = "CentralMoment_1_2"
F_CENTRAL_MOMENT_1_3 = "CentralMoment_1_3"
F_CENTRAL_MOMENT_2_0 = "CentralMoment_2_0"
F_CENTRAL_MOMENT_2_1 = "CentralMoment_2_1"
F_CENTRAL_MOMENT_2_2 = "CentralMoment_2_2"
F_CENTRAL_MOMENT_2_3 = "CentralMoment_2_3"
F_EQUIVALENT_DIAMETER = "EquivalentDiameter"
F_HU_MOMENT_0 = "HuMoment_0"
F_HU_MOMENT_1 = "HuMoment_1"
F_HU_MOMENT_2 = "HuMoment_2"
F_HU_MOMENT_3 = "HuMoment_3"
F_HU_MOMENT_4 = "HuMoment_4"
F_HU_MOMENT_5 = "HuMoment_5"
F_HU_MOMENT_6 = "HuMoment_6"
F_INERTIA_TENSOR_0_0 = "InertiaTensor_0_0"
F_INERTIA_TENSOR_0_1 = "InertiaTensor_0_1"
F_INERTIA_TENSOR_1_0 = "InertiaTensor_1_0"
F_INERTIA_TENSOR_1_1 = "InertiaTensor_1_1"
F_INERTIA_TENSOR_EIGENVALUES_0 = "InertiaTensorEigenvalues_0"
F_INERTIA_TENSOR_EIGENVALUES_1 = "InertiaTensorEigenvalues_1"
F_NORMALIZED_MOMENT_0_0 = "NormalizedMoment_0_0"
F_NORMALIZED_MOMENT_0_1 = "NormalizedMoment_0_1"
F_NORMALIZED_MOMENT_0_2 = "NormalizedMoment_0_2"
F_NORMALIZED_MOMENT_0_3 = "NormalizedMoment_0_3"
F_NORMALIZED_MOMENT_1_0 = "NormalizedMoment_1_0"
F_NORMALIZED_MOMENT_1_1 = "NormalizedMoment_1_1"
F_NORMALIZED_MOMENT_1_2 = "NormalizedMoment_1_2"
F_NORMALIZED_MOMENT_1_3 = "NormalizedMoment_1_3"
F_NORMALIZED_MOMENT_2_0 = "NormalizedMoment_2_0"
F_NORMALIZED_MOMENT_2_1 = "NormalizedMoment_2_1"
F_NORMALIZED_MOMENT_2_2 = "NormalizedMoment_2_2"
F_NORMALIZED_MOMENT_2_3 = "NormalizedMoment_2_3"
F_NORMALIZED_MOMENT_3_0 = "NormalizedMoment_3_0"
F_NORMALIZED_MOMENT_3_1 = "NormalizedMoment_3_1"
F_NORMALIZED_MOMENT_3_2 = "NormalizedMoment_3_2"
F_NORMALIZED_MOMENT_3_3 = "NormalizedMoment_3_3"
F_SPATIAL_MOMENT_0_0 = "SpatialMoment_0_0"
F_SPATIAL_MOMENT_0_1 = "SpatialMoment_0_1"
F_SPATIAL_MOMENT_0_2 = "SpatialMoment_0_2"
F_SPATIAL_MOMENT_0_3 = "SpatialMoment_0_3"
F_SPATIAL_MOMENT_1_0 = "SpatialMoment_1_0"
F_SPATIAL_MOMENT_1_1 = "SpatialMoment_1_1"
F_SPATIAL_MOMENT_1_2 = "SpatialMoment_1_2"
F_SPATIAL_MOMENT_1_3 = "SpatialMoment_1_3"
F_SPATIAL_MOMENT_2_0 = "SpatialMoment_2_0"
F_SPATIAL_MOMENT_2_1 = "SpatialMoment_2_1"
F_SPATIAL_MOMENT_2_2 = "SpatialMoment_2_2"
F_SPATIAL_MOMENT_2_3 = "SpatialMoment_2_3"

"""The non-Zernike features"""
F_STD_2D = [
    F_AREA,
    F_PERIMETER,
    F_MAXIMUM_RADIUS,
    F_MEAN_RADIUS,
    F_MEDIAN_RADIUS,
    F_MIN_FERET_DIAMETER,
    F_MAX_FERET_DIAMETER,
    F_ORIENTATION,
    F_ECCENTRICITY,
    F_FORM_FACTOR,
    F_SOLIDITY,
    F_CONVEX_AREA,
    F_COMPACTNESS,
    F_BBOX_AREA,
]
F_STD_3D = [
    F_VOLUME,
    F_SURFACE_AREA,
    F_CENTER_Z,
    F_BBOX_VOLUME,
    F_MIN_Z,
    F_MAX_Z,
]
F_ADV_2D = [
    F_SPATIAL_MOMENT_0_0,
    F_SPATIAL_MOMENT_0_1,
    F_SPATIAL_MOMENT_0_2,
    F_SPATIAL_MOMENT_0_3,
    F_SPATIAL_MOMENT_1_0,
    F_SPATIAL_MOMENT_1_1,
    F_SPATIAL_MOMENT_1_2,
    F_SPATIAL_MOMENT_1_3,
    F_SPATIAL_MOMENT_2_0,
    F_SPATIAL_MOMENT_2_1,
    F_SPATIAL_MOMENT_2_2,
    F_SPATIAL_MOMENT_2_3,
    F_CENTRAL_MOMENT_0_0,
    F_CENTRAL_MOMENT_0_1,
    F_CENTRAL_MOMENT_0_2,
    F_CENTRAL_MOMENT_0_3,
    F_CENTRAL_MOMENT_1_0,
    F_CENTRAL_MOMENT_1_1,
    F_CENTRAL_MOMENT_1_2,
    F_CENTRAL_MOMENT_1_3,
    F_CENTRAL_MOMENT_2_0,
    F_CENTRAL_MOMENT_2_1,
    F_CENTRAL_MOMENT_2_2,
    F_CENTRAL_MOMENT_2_3,
    F_NORMALIZED_MOMENT_0_0,
    F_NORMALIZED_MOMENT_0_1,
    F_NORMALIZED_MOMENT_0_2,
    F_NORMALIZED_MOMENT_0_3,
    F_NORMALIZED_MOMENT_1_0,
    F_NORMALIZED_MOMENT_1_1,
    F_NORMALIZED_MOMENT_1_2,
    F_NORMALIZED_MOMENT_1_3,
    F_NORMALIZED_MOMENT_2_0,
    F_NORMALIZED_MOMENT_2_1,
    F_NORMALIZED_MOMENT_2_2,
    F_NORMALIZED_MOMENT_2_3,
    F_NORMALIZED_MOMENT_3_0,
    F_NORMALIZED_MOMENT_3_1,
    F_NORMALIZED_MOMENT_3_2,
    F_NORMALIZED_MOMENT_3_3,
    F_HU_MOMENT_0,
    F_HU_MOMENT_1,
    F_HU_MOMENT_2,
    F_HU_MOMENT_3,
    F_HU_MOMENT_4,
    F_HU_MOMENT_5,
    F_HU_MOMENT_6,
    F_INERTIA_TENSOR_0_0,
    F_INERTIA_TENSOR_0_1,
    F_INERTIA_TENSOR_1_0,
    F_INERTIA_TENSOR_1_1,
    F_INERTIA_TENSOR_EIGENVALUES_0,
    F_INERTIA_TENSOR_EIGENVALUES_1,
]
F_ADV_3D = [F_SOLIDITY]
F_STANDARD = [
    F_EXTENT,
    F_EULER_NUMBER,
    F_EQUIVALENT_DIAMETER,
    F_MAJOR_AXIS_LENGTH,
    F_MINOR_AXIS_LENGTH,
    F_CENTER_X,
    F_CENTER_Y,
    F_MIN_X,
    F_MIN_Y,
    F_MAX_X,
    F_MAX_Y,
]


def get_zernike_name(zernike_index):
    """Return the name of a Zernike feature, given a (N,M) 2-tuple

    zernike_index - a 2 element sequence organized as N,M
    """
    return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])


desired_properties = [
    "label",
    "image",
    "area",
    "perimeter",
    "bbox",
    "bbox_area",
    "major_axis_length",
    "minor_axis_length",
    "orientation",
    "centroid",
    "equivalent_diameter",
    "extent",
    "eccentricity",
    "convex_area",
    "solidity",
    "euler_number",
    "inertia_tensor",
    "inertia_tensor_eigvals",
    "moments",
    "moments_central",
    "moments_hu",
    "moments_normalized",
    "solidity",
]


class SPARSE_FIELD(Enum):
    label = "label"
    c = "c"
    t = "t"
    z = "z"
    y = "y"
    x = "x"


class DENSE_AXIS(Enum):
    label_idx = 0
    c = 1
    t = 2
    z = 3
    y = 4
    x = 5


SPARSE_FIELDS = tuple([mem.value for mem in SPARSE_FIELD])
SPARSE_AXES_FIELDS = SPARSE_FIELDS[1:]
DENSE_AXIS_NAMES = tuple([mem.name for mem in DENSE_AXIS])
DENSE_SHAPE_NAMES = DENSE_AXIS_NAMES[1:]


def downsample_labels(labels):
    """
    Convert a 'labels' matrix to the smallest possible integer format
    """

    labels_max = np.max(labels)
    if labels_max < 128:
        return labels.astype(np.int8)
    elif labels_max < 32768:
        return labels.astype(np.int16)
    return labels.astype(np.int32)


def convert_labels_to_dense(labels):
    """
    Convert a 'labels' matrix (e.g. scipy.ndimage.label) to 'dense' matrix
    """

    typed_labels = downsample_labels(labels)

    if labels.ndim == 3:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value
        )
    else:
        expand_axes = (
            DENSE_AXIS.label_idx.value,
            DENSE_AXIS.c.value,
            DENSE_AXIS.t.value,
            DENSE_AXIS.z.value
        )

    return np.expand_dims(typed_labels, axis=expand_axes)


def convert_dense_to_sparse(dense):
    full_shape = dense.shape
    label_dim = full_shape[DENSE_AXIS.label_idx.value]
    dense_shape = tuple(
        [full_shape[DENSE_AXIS[n].value] for n in DENSE_SHAPE_NAMES]
    )

    axes_labels = np.array(SPARSE_AXES_FIELDS)
    axes = axes_labels[np.where(np.array(dense_shape) > 1)]

    compact = np.squeeze(dense)
    if label_dim == 1:
        compact = np.expand_dims(compact, axis=0)

    coords = np.where(compact != 0)
    labels = compact[coords]
    # no longer need the labels dim
    coords = coords[1:]

    if np.max(compact.shape) < 2 ** 16:
        coords_dtype = np.uint16
    else:
        coords_dtype = np.uint32

    if len(labels) > 0:
        max_label = np.max(labels)
        if max_label < 2 ** 8:
            labels_dtype = np.uint8
        elif max_label < 2 ** 16:
            labels_dtype = np.uint16
        else:
            labels_dtype = np.uint32
    else:
        labels_dtype = np.uint8

    dtype = [(axis, coords_dtype) for axis in axes]
    dtype.append((SPARSE_FIELD.label.value, labels_dtype))
    sparse = np._core.records.fromarrays(list(coords) + [labels], dtype=dtype)

    return sparse


def convert_sparse_to_ijv(sparse):
    return np.column_stack([sparse[axis] for axis in (
        SPARSE_FIELD.y.value, SPARSE_FIELD.x.value, SPARSE_FIELD.label.value)
                            ])


def convert_labels_to_ijv(labels):
    dense = convert_labels_to_dense(labels)
    sparse = convert_dense_to_sparse(dense)
    ijv = convert_sparse_to_ijv(sparse)

    return ijv


def get_features(save_cvs_path, list_of_contours, labels_of_contours, image_size=(512, 512)):
    features = []
    mask = np.zeros(image_size, dtype=np.uint8)
    mask_frames = []
    mask_fovs = []
    mask_labels = []

    for i in tqdm(range(len(list_of_contours))):
        contour = list_of_contours[i]
        frame, fov, label_value = labels_of_contours[i]
        j = i + 1
        if j < len(list_of_contours):
            _, next_fov, _ = labels_of_contours[j]
            if fov == next_fov:
                rr, cc = skimage.draw.polygon(contour[:, 0], contour[:, 1], mask.shape)
                mask[rr, cc] = label_value
                mask_frames.append(frame)
                mask_fovs.append(fov)
                mask_labels.append(int(label_value))
                continue

        label_image = mask

        # label_image = skimage.measure.label(mask)
        if len(contour.shape) == 2:
            """Computing the measurements for a single contour"""
            props = skimage.measure.regionprops_table(
                label_image, properties=desired_properties
            )

            formfactor = 4.0 * numpy.pi * props["area"] / props["perimeter"] ** 2
            denom = [max(x, 1) for x in 4.0 * numpy.pi * props["area"]]
            compactness = props["perimeter"] ** 2 / denom

            indexes = list(np.unique(label_image)[1:])
            nobjects = len(indexes)
            max_radius = numpy.zeros(nobjects)
            median_radius = numpy.zeros(nobjects)
            mean_radius = numpy.zeros(nobjects)
            zernike_numbers = centrosome.zernike.get_zernike_indexes(ZERNIKE_N + 1)

            zf = {}
            for n, m in zernike_numbers:
                zf[(n, m)] = numpy.zeros(2)

            for index, mini_image in enumerate(props["image"]):
                # Pad image to assist distance tranform
                mini_image = numpy.pad(mini_image, 1)
                distances = scipy.ndimage.distance_transform_edt(mini_image)
                max_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.maximum(distances, mini_image)
                )
                mean_radius[index] = centrosome.cpmorphology.fixup_scipy_ndimage_result(
                    scipy.ndimage.mean(distances, mini_image)
                )
                median_radius[index] = centrosome.cpmorphology.median_of_labels(
                    distances, mini_image.astype("int"), [1]
                )
            #
            # Zernike features
            #
            zf_l = centrosome.zernike.zernike(
                zernike_numbers, label_image, indexes
            )
            for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                zf[(n, m)] = z

            chulls, chull_counts = centrosome.cpmorphology.convex_hull_ijv(
                convert_labels_to_ijv(label_image), indexes
            )
            #
            # Feret diameter
            #
            min_feret_diameter, max_feret_diameter = centrosome.cpmorphology.feret_diameter(chulls, chull_counts, indexes)

            feature_dict = {
                "Frame": mask_frames,
                "Fov": mask_fovs,
                "Label": mask_labels,
                F_AREA: props["area"],
                F_PERIMETER: props["perimeter"],
                F_MAJOR_AXIS_LENGTH: props["major_axis_length"],
                F_MINOR_AXIS_LENGTH: props["minor_axis_length"],
                F_ECCENTRICITY: props["eccentricity"],
                F_ORIENTATION: props["orientation"] * (180 / numpy.pi),
                F_CENTER_X: props["centroid-1"],
                F_CENTER_Y: props["centroid-0"],
                F_BBOX_AREA: props["bbox_area"],
                F_MIN_X: props["bbox-1"],
                F_MAX_X: props["bbox-3"],
                F_MIN_Y: props["bbox-0"],
                F_MAX_Y: props["bbox-2"],
                F_FORM_FACTOR: formfactor,
                F_EXTENT: props["extent"],
                F_SOLIDITY: props["solidity"],
                F_COMPACTNESS: compactness,
                F_EULER_NUMBER: props["euler_number"],
                F_MAXIMUM_RADIUS: max_radius,
                F_MEAN_RADIUS: mean_radius,
                F_MEDIAN_RADIUS: median_radius,
                F_CONVEX_AREA: props["convex_area"],
                F_MIN_FERET_DIAMETER: min_feret_diameter,
                F_MAX_FERET_DIAMETER: max_feret_diameter,
                F_EQUIVALENT_DIAMETER: props["equivalent_diameter"],
                F_SPATIAL_MOMENT_0_0: props["moments-0-0"],
                F_SPATIAL_MOMENT_0_1: props["moments-0-1"],
                F_SPATIAL_MOMENT_0_2: props["moments-0-2"],
                F_SPATIAL_MOMENT_0_3: props["moments-0-3"],
                F_SPATIAL_MOMENT_1_0: props["moments-1-0"],
                F_SPATIAL_MOMENT_1_1: props["moments-1-1"],
                F_SPATIAL_MOMENT_1_2: props["moments-1-2"],
                F_SPATIAL_MOMENT_1_3: props["moments-1-3"],
                F_SPATIAL_MOMENT_2_0: props["moments-2-0"],
                F_SPATIAL_MOMENT_2_1: props["moments-2-1"],
                F_SPATIAL_MOMENT_2_2: props["moments-2-2"],
                F_SPATIAL_MOMENT_2_3: props["moments-2-3"],
                F_CENTRAL_MOMENT_0_0: props["moments_central-0-0"],
                F_CENTRAL_MOMENT_0_1: props["moments_central-0-1"],
                F_CENTRAL_MOMENT_0_2: props["moments_central-0-2"],
                F_CENTRAL_MOMENT_0_3: props["moments_central-0-3"],
                F_CENTRAL_MOMENT_1_0: props["moments_central-1-0"],
                F_CENTRAL_MOMENT_1_1: props["moments_central-1-1"],
                F_CENTRAL_MOMENT_1_2: props["moments_central-1-2"],
                F_CENTRAL_MOMENT_1_3: props["moments_central-1-3"],
                F_CENTRAL_MOMENT_2_0: props["moments_central-2-0"],
                F_CENTRAL_MOMENT_2_1: props["moments_central-2-1"],
                F_CENTRAL_MOMENT_2_2: props["moments_central-2-2"],
                F_CENTRAL_MOMENT_2_3: props["moments_central-2-3"],
                F_NORMALIZED_MOMENT_0_0: props["moments_normalized-0-0"],
                F_NORMALIZED_MOMENT_0_1: props["moments_normalized-0-1"],
                F_NORMALIZED_MOMENT_0_2: props["moments_normalized-0-2"],
                F_NORMALIZED_MOMENT_0_3: props["moments_normalized-0-3"],
                F_NORMALIZED_MOMENT_1_0: props["moments_normalized-1-0"],
                F_NORMALIZED_MOMENT_1_1: props["moments_normalized-1-1"],
                F_NORMALIZED_MOMENT_1_2: props["moments_normalized-1-2"],
                F_NORMALIZED_MOMENT_1_3: props["moments_normalized-1-3"],
                F_NORMALIZED_MOMENT_2_0: props["moments_normalized-2-0"],
                F_NORMALIZED_MOMENT_2_1: props["moments_normalized-2-1"],
                F_NORMALIZED_MOMENT_2_2: props["moments_normalized-2-2"],
                F_NORMALIZED_MOMENT_2_3: props["moments_normalized-2-3"],
                F_NORMALIZED_MOMENT_3_0: props["moments_normalized-3-0"],
                F_NORMALIZED_MOMENT_3_1: props["moments_normalized-3-1"],
                F_NORMALIZED_MOMENT_3_2: props["moments_normalized-3-2"],
                F_NORMALIZED_MOMENT_3_3: props["moments_normalized-3-3"],
                F_HU_MOMENT_0: props["moments_hu-0"],
                F_HU_MOMENT_1: props["moments_hu-1"],
                F_HU_MOMENT_2: props["moments_hu-2"],
                F_HU_MOMENT_3: props["moments_hu-3"],
                F_HU_MOMENT_4: props["moments_hu-4"],
                F_HU_MOMENT_5: props["moments_hu-5"],
                F_HU_MOMENT_6: props["moments_hu-6"],
                F_INERTIA_TENSOR_0_0: props["inertia_tensor-0-0"],
                F_INERTIA_TENSOR_0_1: props["inertia_tensor-0-1"],
                F_INERTIA_TENSOR_1_0: props["inertia_tensor-1-0"],
                F_INERTIA_TENSOR_1_1: props["inertia_tensor-1-1"],
                F_INERTIA_TENSOR_EIGENVALUES_0: props[
                    "inertia_tensor_eigvals-0"
                ],
                F_INERTIA_TENSOR_EIGENVALUES_1: props[
                    "inertia_tensor_eigvals-1"
                ],
            }
            feature_dict.update(
                {
                    get_zernike_name((n, m)): zf[(n, m)]
                    for n, m in zernike_numbers
                }
            )
            feature_dict[F_SOLIDITY] = props["solidity"]

            features.append(pd.DataFrame(feature_dict))

            mask = np.zeros(image_size, dtype=np.uint8)
            mask_frames = []
            mask_fovs = []
            mask_labels = []

    df = pd.concat(features, axis=0)
    df.to_csv(save_cvs_path, index=False)

