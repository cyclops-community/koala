
class Site(object):
    def __init__(self, data=None):
        if data == None:
            spin_up = np.reshape(
                np.array([1, 0], dtype=complex), 
                [1, 1, 1, 1, 2]
                )
            self.data = np.copy(spin_up)
        else:
            self.data = data