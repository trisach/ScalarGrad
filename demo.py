from micronet import MLP

n = MLP(3,[4,4,1])

xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]

ys = [1.0,-1.0,-1.0,1]
for i in range(50):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) **2 for ygt,yout in zip(ys,ypred))

    for p in n.parameters():
        p.grad = 0
    loss.backward()

    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(f'iter {i}, loss {loss.data}')   

finalpred = [n(x) for x in xs]
print(f'Predicted Output : {[i.data for i in finalpred]}\nGround Truth : {ys}')
