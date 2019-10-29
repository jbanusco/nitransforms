"""Read/write ITK transforms."""
import numpy as np
from scipy.io import savemat as _save_mat
from nibabel.affines import from_matvec
from .base import StringBasedStruct, _read_mat, TransformFileError

LPS = np.diag([-1, -1, 1, 1])


class ITKLinearTransform(StringBasedStruct):
    """A string-based structure for ITK linear transforms."""

    template_dtype = np.dtype([
        ('type', 'i4'),
        ('index', 'i4'),
        ('parameters', 'f4', (4, 4)),
        ('offset', 'f4', 3),  # Center of rotation
    ])
    dtype = template_dtype
    # files_types = (('string', '.tfm'), ('binary', '.mat'))
    # valid_exts = ('.tfm', '.mat')

    def __init__(self, parameters=None, offset=None):
        """Initialize with default offset and index."""
        super().__init__()
        self.structarr['index'] = 0
        self.structarr['offset'] = offset or [0, 0, 0]
        self.structarr['parameters'] = np.eye(4)
        if parameters is not None:
            self.structarr['parameters'] = parameters

    def __str__(self):
        """Generate a string representation."""
        sa = self.structarr
        lines = [
            '#Transform {:d}'.format(sa['index']),
            'Transform: AffineTransform_float_3_3',
            'Parameters: {}'.format(' '.join(
                ['%g' % p
                 for p in sa['parameters'][:3, :3].reshape(-1).tolist() +
                 sa['parameters'][:3, 3].tolist()])),
            'FixedParameters: {:g} {:g} {:g}'.format(*sa['offset']),
            '',
        ]
        return '\n'.join(lines)

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        if str(filename).endswith('.mat'):
            sa = self.structarr
            affine = np.array(np.hstack((
                sa['parameters'][:3, :3].reshape(-1),
                sa['parameters'][:3, 3]))[..., np.newaxis], dtype='f4')
            fixed = np.array(sa['offset'][..., np.newaxis], dtype='f4')
            mdict = {
                'AffineTransform_float_3_3': affine,
                'fixed': fixed,
            }
            _save_mat(str(filename), mdict, format='4')
            return

        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self):
        """Return a nitransforms internal RAS+ matrix."""
        sa = self.structarr
        matrix = sa['parameters']
        offset = sa['offset']
        c_neg = from_matvec(np.eye(3), offset * -1.0)
        c_pos = from_matvec(np.eye(3), offset)
        return LPS.dot(c_pos.dot(matrix.dot(c_neg.dot(LPS))))

    def to_string(self, banner=None):
        """Convert to a string directly writeable to file."""
        string = '%s'

        if banner is None:
            banner = self.structarr['index'] == 0

        if banner:
            string = '#Insight Transform File V1.0\n%s'
        return string % self

    @classmethod
    def from_binary(cls, byte_stream, index=0):
        """Read the struct from a matlab binary file."""
        mdict = _read_mat(byte_stream)
        return cls.from_matlab_dict(mdict, index=index)

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        if fileobj.name.endswith('.mat'):
            return cls.from_binary(fileobj)
        return cls.from_string(fileobj.read())

    @classmethod
    def from_matlab_dict(cls, mdict, index=0):
        """Read the struct from a matlab dictionary."""
        tf = cls()
        sa = tf.structarr

        sa['index'] = index
        parameters = np.eye(4, dtype='f4')
        parameters[:3, :3] = mdict['AffineTransform_float_3_3'][:-3].reshape((3, 3))
        parameters[:3, 3] = mdict['AffineTransform_float_3_3'][-3:].flatten()
        sa['parameters'] = parameters
        sa['offset'] = mdict['fixed'].flatten()
        return tf

    @classmethod
    def from_ras(cls, ras, index=0):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        tf = cls()
        sa = tf.structarr
        sa['index'] = index
        sa['parameters'] = LPS.dot(ras.dot(LPS))
        return tf

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        tf = cls()
        sa = tf.structarr
        lines = [l for l in string.splitlines()
                 if l.strip()]
        assert lines[0][0] == '#'
        if lines[1][0] == '#':
            lines = lines[1:]  # Drop banner with version

        parameters = np.eye(4, dtype='f4')
        sa['index'] = int(lines[0][lines[0].index('T'):].split()[1])
        sa['offset'] = np.genfromtxt([lines[3].split(':')[-1].encode()],
                                     dtype=cls.dtype['offset'])
        vals = np.genfromtxt([lines[2].split(':')[-1].encode()],
                             dtype='f4')
        parameters[:3, :3] = vals[:-3].reshape((3, 3))
        parameters[:3, 3] = vals[-3:]
        sa['parameters'] = parameters
        return tf


class ITKLinearTransformArray(StringBasedStruct):
    """A string-based structure for series of ITK linear transforms."""

    template_dtype = np.dtype([('nxforms', 'i4')])
    dtype = template_dtype
    _xforms = None

    def __init__(self,
                 xforms=None,
                 binaryblock=None,
                 endianness=None,
                 check=True):
        """Initialize with (optionally) a list of transforms."""
        super().__init__(binaryblock, endianness, check)
        self.xforms = [ITKLinearTransform(parameters=mat)
                       for mat in xforms or []]

    @property
    def xforms(self):
        """Get the list of internal ITKLinearTransforms."""
        return self._xforms

    @xforms.setter
    def xforms(self, value):
        self._xforms = list(value)

        # Update indexes
        for i, val in enumerate(self.xforms):
            val['index'] = i

    def __getitem__(self, idx):
        """Allow dictionary access to the transforms."""
        if idx == 'xforms':
            return self._xforms
        if idx == 'nxforms':
            return len(self._xforms)
        raise KeyError(idx)

    def to_filename(self, filename):
        """Store this transform to a file with the appropriate format."""
        if str(filename).endswith('.mat'):
            raise TransformFileError("Please use the ITK's new .h5 format.")

        with open(str(filename), 'w') as f:
            f.write(self.to_string())

    def to_ras(self):
        """Return a nitransforms' internal RAS matrix."""
        return np.stack([xfm.to_ras() for xfm in self.xforms])

    def to_string(self):
        """Convert to a string directly writeable to file."""
        strings = []
        for i, xfm in enumerate(self.xforms):
            xfm.structarr['index'] = i
            strings.append(xfm.to_string())
        return '\n'.join(strings)

    @classmethod
    def from_binary(cls, byte_stream):
        """Read the struct from a matlab binary file."""
        raise TransformFileError("Please use the ITK's new .h5 format.")

    @classmethod
    def from_fileobj(cls, fileobj, check=True):
        """Read the struct from a file object."""
        if fileobj.name.endswith('.mat'):
            return cls.from_binary(fileobj)
        return cls.from_string(fileobj.read())

    @classmethod
    def from_ras(cls, ras):
        """Create an ITK affine from a nitransform's RAS+ matrix."""
        _self = cls()
        _self.xforms = [ITKLinearTransform.from_ras(ras[i, ...], i)
                        for i in range(ras.shape[0])]
        return _self

    @classmethod
    def from_string(cls, string):
        """Read the struct from string."""
        _self = cls()
        lines = [l.strip() for l in string.splitlines()
                 if l.strip()]

        if lines[0][0] != '#' or 'Insight Transform File V1.0' not in lines[0]:
            raise ValueError('Unknown Insight Transform File format.')

        string = '\n'.join(lines[1:])
        for xfm in string.split('#')[1:]:
            _self.xforms.append(ITKLinearTransform.from_string(
                '#%s' % xfm))
        return _self