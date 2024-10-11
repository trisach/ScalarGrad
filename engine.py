import math
class Value:
    def __init__(self,data,_children = (),_op='',label = ''):
        self.data = data
        self._backward = lambda : None  #-->for leaf nodes
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        
    def __repr__(self):
        return f"Value (data = {self.data})"

    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
             self.grad += 1.0 * out.grad                                
             other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self,other):
        return self+other
        
    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self,other):
        return self*other

    def __pow__(self,other):
        assert isinstance(other,(int,float)), "Only accepts int or float for now"
        out = Value(self.data**other,(self, ),f'**{other}')
        def _backward():
            self.grad += other* self.data**(other-1) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self,other):
        return self + (-other)
        
    def __truediv__(self,other):
        return self * other**-1
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self, ),'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t,(self, ),'tanh')
        
        def _backward():
            self.grad += (1 - t**2) * out.grad             #local derivative * the 'chained' derivative
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        # val = 0 if 0>x else x
        out = Value(max(0,x),(self, ),'relu')
        def _backward():
            self.grad += self.data * out.grad

        return out 
    
    def sigmoid(self):
        x = self.data
        out = Value(1/(1+math.exp(-x)),(self, ),'sigmoid')
        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


