from numpy._utils import set_module
from numpy.core import numeric as sb
from numpy.core import numerictypes as nt
from numpy.core.arrayprint import _get_legacy_print_mode
from numpy.core.records import format_parser

# All of the functions allow formats to be a dtype
__all__ = [
    'record', 'recarray', 'format_parser',
    'fromarrays', 'fromrecords', 'fromstring', 'fromfile', 'array',
]


ndarray = sb.ndarray

# formats regular expression
# allows multidimensional spec with a tuple syntax in front
# of the letter code '(2,3)f4' and ' (  2 ,  3  )  f4  '
# are equally allowed

numfmt = nt.sctypeDict


class record(nt.void):
    """A data-type scalar that allows field access as attribute lookup.
    """

    # manually set name and module so that this class's type shows up
    # as numpy.record when printed
    __name__ = 'record'
    __module__ = 'numpy'

    def __repr__(self):
        if _get_legacy_print_mode() <= 113:
            return self.__str__()
        return super().__repr__()

    def __str__(self):
        if _get_legacy_print_mode() <= 113:
            return str(self.item())
        return super().__str__()

    def __getattribute__(self, attr):
        if attr in ('setfield', 'getfield', 'dtype'):
            return nt.void.__getattribute__(self, attr)
        try:
            return nt.void.__getattribute__(self, attr)
        except AttributeError:
            pass
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            obj = self.getfield(*res[:2])
            # if it has fields return a record,
            # otherwise return the object
            try:
                dt = obj.dtype
                dt_names = dt.names
            except AttributeError:
                #happens if field is Object type
                return obj
            if dt_names is not None:
                return obj.view((self.__class__, dt))
            return obj
        else:
            raise AttributeError("'record' object has no "
                    "attribute '%s'" % attr)

    def __setattr__(self, attr, val):
        if attr in ('setfield', 'getfield', 'dtype'):
            raise AttributeError("Cannot set '%s' attribute" % attr)
        fielddict = nt.void.__getattribute__(self, 'dtype').fields
        res = fielddict.get(attr, None)
        if res:
            return self.setfield(val, *res[:2])
        else:
            if getattr(self, attr, None):
                return nt.void.__setattr__(self, attr, val)
            else:
                raise AttributeError("'record' object has no "
                        "attribute '%s'" % attr)

    def __getitem__(self, indx):
        obj = nt.void.__getitem__(self, indx)

        # copy behavior of record.__getattribute__,
        if isinstance(obj, nt.void) and obj.dtype.names is not None:
            return obj.view((self.__class__, obj.dtype))
        else:
            # return a single element
            return obj

    def pprint(self):
        """Pretty-print all fields."""
        # pretty-print all fields
        names = self.dtype.names
        maxlen = max(len(name) for name in names)
        fmt = '%% %ds: %%s' % maxlen
        rows = [fmt % (name, getattr(self, name)) for name in names]
        return "\n".join(rows)

# The recarray is almost identical to a standard array (which supports
#   named fields already)  The biggest difference is that it can use
#   attribute-lookup to find the fields and it is constructed using
#   a record.

# If byteorder is given it forces a particular byteorder on all
#  the fields (and any subfields)

class recarray(ndarray):
    # manually set name and module so that this class's type shows
    # up as "numpy.recarray" when printed
    __name__ = 'recarray'
    __module__ = 'numpy'

    def __new__(subtype, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False, order='C'):

        if dtype is not None:
            descr = sb.dtype(dtype)
        else:
            descr = format_parser(formats, names, titles, aligned, byteorder).dtype

        if buf is None:
            self = ndarray.__new__(subtype, shape, (record, descr), order=order)
        else:
            self = ndarray.__new__(subtype, shape, (record, descr),
                                      buffer=buf, offset=offset,
                                      strides=strides, order=order)
        return self

    def __array_finalize__(self, obj):
        if self.dtype.type is not record and self.dtype.names is not None:
            # if self.dtype is not np.record, invoke __setattr__ which will
            # convert it to a record if it is a void dtype.
            self.dtype = self.dtype

    def __getattribute__(self, attr):
        # See if ndarray has this attr, and return it if so. (note that this
        # means a field with the same name as an ndarray attr cannot be
        # accessed by attribute).
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass

        # look for a field with this name
        fielddict = ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError("recarray has no attribute %s" % attr) from e
        obj = self.getfield(*res)

        # At this point obj will always be a recarray, since (see
        # PyArray_GetField) the type of obj is inherited. Next, if obj.dtype is
        # non-structured, convert it to an ndarray. Then if obj is structured
        # with void type convert it to the same dtype.type (eg to preserve
        # numpy.record type if present), since nested structured fields do not
        # inherit type. Don't do this for non-void structures though.
        if obj.dtype.names is not None:
            if issubclass(obj.dtype.type, nt.void):
                return obj.view(dtype=(self.dtype.type, obj.dtype))
            return obj
        else:
            return obj.view(ndarray)

    # Save the dictionary.
    # If the attr is a field name and not in the saved dictionary
    # Undo any "setting" of the attribute and do a setfield
    # Thus, you can't create attributes on-the-fly that are field names.
    def __setattr__(self, attr, val):

        # Automatically convert (void) structured types to records
        # (but not non-void structures, subarrays, or non-structured voids)
        if attr == 'dtype' and issubclass(val.type, nt.void) and val.names is not None:
            val = sb.dtype((record, val))

        newattr = attr not in self.__dict__
        try:
            ret = object.__setattr__(self, attr, val)
        except Exception:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                raise
        else:
            fielddict = ndarray.__getattribute__(self, 'dtype').fields or {}
            if attr not in fielddict:
                return ret
            if newattr:
                # We just added this one or this setattr worked on an
                # internal attribute.
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                "record array has no attribute %s" % attr
            ) from e
        return self.setfield(val, *res)

    def __getitem__(self, indx):
        obj = super().__getitem__(indx)

        # copy behavior of getattr, except that here
        # we might also be returning a single element
        if isinstance(obj, ndarray):
            if obj.dtype.names is not None:
                obj = obj.view(type(self))
                if issubclass(obj.dtype.type, nt.void):
                    return obj.view(dtype=(self.dtype.type, obj.dtype))
                return obj
            else:
                return obj.view(type=ndarray)
        else:
            # return a single element
            return obj

    def __repr__(self):

        repr_dtype = self.dtype
        if self.dtype.type is record or not issubclass(self.dtype.type, nt.void):
            # If this is a full record array (has numpy.record dtype),
            # or if it has a scalar (non-void) dtype with no records,
            # represent it using the rec.array function. Since rec.array
            # converts dtype to a numpy.record for us, convert back
            # to non-record before printing
            if repr_dtype.type is record:
                repr_dtype = sb.dtype((nt.void, repr_dtype))
            prefix = "rec.array("
            fmt = 'rec.array(%s,%sdtype=%s)'
        else:
            # otherwise represent it using np.array plus a view
            # This should only happen if the user is playing
            # strange games with dtypes.
            prefix = "array("
            fmt = 'array(%s,%sdtype=%s).view(numpy.recarray)'

        # get data/shape string. logic taken from numeric.array_repr
        if self.size > 0 or self.shape == (0,):
            lst = sb.array2string(
                self, separator=', ', prefix=prefix, suffix=',')
        else:
            # show zero-length shape unless it is (0,)
            lst = "[], shape=%s" % (repr(self.shape),)

        lf = '\n'+' '*len(prefix)
        if _get_legacy_print_mode() <= 113:
            lf = ' ' + lf  # trailing space
        return fmt % (lst, lf, repr_dtype)

    def field(self, attr, val=None):
        if isinstance(attr, int):
            names = ndarray.__getattribute__(self, 'dtype').names
            attr = names[attr]

        fielddict = ndarray.__getattribute__(self, 'dtype').fields

        res = fielddict[attr][:2]

        if val is None:
            obj = self.getfield(*res)
            if obj.dtype.names is not None:
                return obj
            return obj.view(ndarray)
        else:
            return self.setfield(val, *res)




