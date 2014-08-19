import larank

print "world"

class cc(object):
    def __init__(self):
        self.__x = 5
        self.__y = 3
        self.z = 4
    def abc(self):
        print 'ok'

class made(object):
    def test(self):
        ok = cc()
        print ok.z
        print 1e-3

c = made()
c.test()
a = larank.SVM_model()
a.test()