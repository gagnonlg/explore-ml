# Deep learning textbook, sec. 8.5

def SGD(theta, epsilon):
    while not stopped():
        g = sum(
            [grad(loss(f(xi, theta), yi), theta)
             for xi, yi in minibatch()]
        )
        theta -= epsilon * g


def SGD_momentum(theta, velo, epsilon, alpha):
    while not stopped():
        g = sum(
            [grad(loss(f(xi, theta), yi), theta)
             for xi, yi in minibatch()]
        )
        velo = alpha * velo - epsilon * g
        theta += velo

def SGD_momentum_nesterov(theta, velo, epsilon, alpha):
    while not stopped():
        theta_tilde = theta + alpha * velo
        g = sum(
            [grad(loss(f(xi, theta_tilde), yi), theta_tilde)
             for xi, yi in minibatch()]
        )
        velo = alpha * velo - epsilon * g
        theta += velo

def AdaGrad(theta, epsilon, delta):
    r = zeros()
    while not stopped():
        g = sum(
            [grad(loss(f(xi, theta), yi), theta)
             for xi, yi in minibatch()]
        )
        r += g ** 2 # element-wise
        theta -= epsilon * g / (delta + sqrt(r)) # element-wise

def RMSProp(theta, epsilon, rho, delta):
    r = zeros()
    while not stopped():
        g = sum(
            [grad(loss(f(xi, theta), yi), theta)
             for xi, yi in minibatch()]
        )
        r = rho * r + (1 - rho) * (g ** 2) # element-wise
        theta -= epsilon * g / (sqrt(delta + r)) # element-wise

def RMSProp_nesterov(theta, epsilon, rho, alpha):
    r = zeros()
    while not stopped():
        theta_tilde = theta + alpha * velo
        g = sum(
            [grad(loss(f(xi, theta_tilde), yi), theta_tilde)
             for xi, yi in minibatch()]
        )
        r = rho * r + (1 - rho) * (g ** 2) # element-wise
        v = alpha * v - epsilon * g / (sqrt(delta + r)) # element-wise
        theta += v

def Adam(theta, epsilon=0.001, rho1=0.9, rho2=0.999, delta=10e-8):
    r1 = zeros()
    r2 = zeros()
    t = 0
    while not stopped():
        g = sum(
            [grad(loss(f(xi, theta), yi), theta)
             for xi, yi in minibatch()]
        )
        t += 1
        r1 = rho1 * r1 + (1 - rho1) * g # element-wise
        r1 /= (1 - rho1**t)
        r2 = rho2 * r2 + (1 - rho2) * (g ** 2) # element-wise
        t2 /= (1 - rho2**t)
        theta -= epsilon * r1 / (delta + sqrt(r2)) # element-wise
