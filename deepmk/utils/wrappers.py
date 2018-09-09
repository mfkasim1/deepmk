
class ObjectsWrapper(list):
    """
    Wrap a multiple object to behave like a single object.
    If a property is called, then it returns a list of the property's values for
    each object. If a function is called, then it applies the function to all
    the contained objects.
    """
    def __init__(self, objs):
        super(ObjectsWrapper, self).__init__(objs)

    def __getattr__(self, name):
        attr = getattr(self[0], name)
        if callable(attr):
            def func(*args, **kwargs):
                return [getattr(m, name)(*args, **kwargs) \
                        for m in self]

            return func
        else:
            return [getattr(m, name) for m in self]
