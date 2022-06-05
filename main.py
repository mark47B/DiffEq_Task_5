import math
from math import cos, sin, sqrt
import numpy as np
from matplotlib import pyplot as plt

A = 1 / 14
B = 1 / 15
c = 1 / 14
a21 = c
b2 = 1.0 / (2.0 * c)  # b2 = 7.0
b1 = 1 - b2  # b1 = -6.0
y1 = B * math.pi
y2 = A * math.pi
(x0, xk) = (0, math.pi)
s = 2
eps = 1e-4

y1_pi_true = 0.2544987421377396
y2_pi_true = 0.1756247987851049


def find_c1_c2(x, b1, b2):
    k = x / math.sqrt(210)
    A_ = [[cos(k), sin(k)], [-14 / sqrt(210) * sin(k), 14 / sqrt(210) * cos(k)]]
    b_ = [[b1], [b2]]
    ans = np.linalg.solve(A_, b_)
    c1 = ans[0][0]
    c2 = ans[1][0]
    return (c1, c2)


c1, c2 = find_c1_c2(0, math.pi / 15, math.pi / 14)

print(math.pi / 15, math.pi / 14 * sqrt(15 / 14))
print(c1, c2)


def general_solution(x, c1, c2):
    y1 = c1 * cos(x / sqrt(210)) + c2 * sin(x / sqrt(210))
    y2 = -14 / sqrt(210) * c1 * sin(x / sqrt(210)) + 14 / sqrt(210) * c2 * cos(x / sqrt(210))
    return (y1, y2)


general_solution(math.pi, c1, c2)

delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
h = pow(eps / delta, 1 / 3)
print(f'Начальный шаг = {h}')


def step(y1, y2, h):
    k11 = h * A * y2
    k21 = h * (-B) * y1

    k12 = h * A * (y2 + a21 * k21)
    k22 = h * (-B) * (y1 + a21 * k11)

    return (y1 + b1 * k11 + b2 * k12, y2 + b1 * k21 + b2 * k22)


error = 1
errors = []

differences = []
points = []

s = 2
delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
h = pow(eps / delta, 1 / 3)

print(f'Начальный шаг = {h}')

while error > eps:
    # for i in range(10): # Поменять на while error > eps когда разберемся с погрешностью
    x0 = 0
    xk = x0
    y_next1 = y1
    y_next2 = y2
    # сначала с шагом h1 = h
    # print(f'============================Epoch = {i}============================')
    h1 = h

    while xk + h1 < math.pi:
        (y_next1, y_next2) = step(y_next1, y_next2, h1)
        xk += h1
        true_error_1 = y_next1 - general_solution(xk)[0]
        true_erorr_2 = y_next2 - general_solution(xk)[1]
        differences.append(math.sqrt(true_error_1 ** 2 + true_erorr_2 ** 2))
        points.append(xk)

        # print(xk)
        # print(y_next1, y_next2)

        if xk + h1 >= math.pi:
            h_final = math.pi - xk
            (y_next1, y_next2) = step(y_next1, y_next2, h_final)
            xk += h_final
            points.append(xk)
            true_error_1 = y_next1 - general_solution(xk)[0]
            true_erorr_2 = y_next2 - general_solution(xk)[1]
            differences.append(math.sqrt(true_error_1 ** 2 + true_erorr_2 ** 2))
            # print(xk)
            # print(y_next1, y_next2)

    res_h1_0, res_h1_1 = (y_next1, y_next2)  # y с чертой (-y)
    print(f'final estimation with h = {h1}, results = {xk, res_h1_0, res_h1_1}')

    xk = x0
    y_next1 = y1
    y_next2 = y2
    h2 = h / 2

    while xk + h2 < math.pi:
        (y_next1, y_next2) = step(y_next1, y_next2, h2)
        xk += h2
        # print(xk)
        # print(y_next1, y_next2)

        if xk + h2 >= math.pi:
            h_final = math.pi - xk
            (y_next1, y_next2) = step(y_next1, y_next2, h_final)
            xk += h_final
            # print(xk)
            # print(y_next1, y_next2)

    res_h2_0, res_h2_1 = (y_next1, y_next2)  # y с волной (~y)
    print(f'final estimation with h = {h2}, results = {xk, res_h2_0, res_h2_1}')
    # -y - h
    # ~y - h/2
    err1 = (res_h2_0 - res_h1_0) / (pow(2, s) - 1)  # ~Ri0
    err2 = (res_h2_1 - res_h1_1) / (pow(2, s) - 1)  # ~Ri1
    error = math.sqrt(err1 ** 2 + err2 ** 2)

    print(error)
    errors.append(error)
    print("Absolute error = ", y1_pi_true - res_h1_0, y2_pi_true - res_h1_1)

    h /= 2

    # print format(floatvalue, '.4f')

    tol1 = 1e-5
    tol2 = tol1 / pow(2, s)
    tol3 = tol2 / pow(2, s + 1)
    # tol3 < tol2 < tol1 = 10^(-5)

    xk = x0
    eps = 1e-04
    s = 2

    delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
    h = pow(eps / delta, 1 / (s + 1)) / 2
    h_i = h

    h_list = []
    points = []

    print('eps = {:.6f}'.format(eps))
    print('h = {:.6f}'.format(h))

    print("Starting point y1_0, y2_0: {:.6f}, {:.6f}".format(y1, y2))
    (y1_k, y2_k, y1_k_2, y2_k_2) = (y1, y2, y1, y2)

    while xk + h_i < math.pi:
        y1_old = y1_k
        y2_old = y2_k
        (y1new, y2new) = step(y1_k, y2_k, h_i)  # шаг с шагом h

        h_i2 = h_i / 2
        (y1newdva, y2newdva) = step(y1_k, y2_k, h_i2)  # шаг с шагом h/2
        (y1_k_2, y2_k_2) = step(y1newdva, y2newdva, h_i2)  # шаг с шагом h/2

        (y1_k, y2_k) = (y1new, y2new)  # новые найденные знач-я y1_k, y2_k

        err1 = (y1_k_2 - y1_k) / (1 - pow(2, -s))  # ~Ri0
        err2 = (y2_k_2 - y2_k) / (1 - pow(2, -s))  # ~Ri1
        error = math.sqrt(err1 ** 2 + err2 ** 2)

        if (error > eps * pow(2, s)):
            h_i /= 2
            print('h_i has been decreased by 2')
            y1_k = y1_old
            y2_k = y2_old
            y1_k_2 = y1_old
            y2_k_2 = y2_old

        elif (error > eps and error <= eps * pow(2, s)):
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            points.append(xk)
            h_list.append(h_i)
            y1_k = y1_k_2
            y2_k = y2_k_2
            xk += h_i
            h_i /= 2
            # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
            # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')
            # y1real, y2real - трушные значения решения в точке Pi (из общего решения руками найти)

        elif (error >= eps / pow(2, s + 1) and error <= eps):
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            points.append(xk)
            h_list.append(h_i)
            y1_k = y1new
            y2_k = y2new
            xk += h_i
            h_i = h_i
            # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
            # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')

        else:
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            points.append(xk)
            h_list.append(h_i)
            y1_k = y1new
            y2_k = y2new
            xk += h_i
            h_i = h_i * 2

            # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
            # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')

        print('y1_k = {:.8f}, y2_k = {:.8f}'.format(y1_k, y2_k))

    # now final step to reach xk = pi
    print('y1_k, y2_k before last step = ', y1_k, y2_k)
    h_final = math.pi - xk
    print('xk = ', xk)
    print('h_final = ', h_final)
    h_list.append(h_final)
    points.append(math.pi)
    (y1new, y2new) = step(y1_k, y2_k, h_final)
    (y1_k, y2_k) = (y1new, y2new)
    print(f'final result at xk = {xk + h_final}: {y1_k, y2_k}')

    print("Absolute error = ", y1_pi_true - y1_k, y2_pi_true - y2_k)


def step_k1k2k3(y1, y2, h):
    k11 = h * A * y2
    k21 = h * (-B) * y1

    k12 = h * A * (y2 + 1 / 2 * k21)
    k22 = h * (-B) * (y1 + 1 / 2 * k11)

    k31 = h * A * (y2 - k21 + 2 * k22)
    k32 = h * (-B) * (y1 - k11 + 2 * k12)

    return (y1 + 1 / 6 * (k11 + 4 * k12 + k31), y2 + 1 / 6 * (k21 + 4 * k22 + k32))


error = 1
errors = []

differences_k1k2k3 = []
points_k1k2k3 = []

s = 3
delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
h = pow(eps / delta, 1 / 3)
print(f'Начальный шаг = {h}')

while error > eps:  # Поменять на while error > eps когда разберемся с погрешностью
    # for i in range(10):
    x0 = 0
    xk = x0
    y_next1 = y1
    y_next2 = y2
    # сначала с шагом h1 = h
    # print(f'============================Epoch = {i}============================')
    h1 = h

    while xk + h1 < math.pi:
        (y_next1, y_next2) = step_k1k2k3(y_next1, y_next2, h1)
        xk += h1
        points_k1k2k3.append(xk)
        true_error_1 = y_next1 - general_solution(xk)[0]
        true_erorr_2 = y_next2 - general_solution(xk)[1]
        differences_k1k2k3.append(math.sqrt(true_error_1 ** 2 + true_erorr_2 ** 2))

        print('xk = ', xk)
        print(y_next1, y_next2)

        if xk + h1 >= math.pi:
            h_final = math.pi - xk
            (y_next1, y_next2) = step_k1k2k3(y_next1, y_next2, h_final)
            xk += h_final
            points_k1k2k3.append(xk)
            true_error_1 = y_next1 - general_solution(xk)[0]
            true_erorr_2 = y_next2 - general_solution(xk)[1]
            differences_k1k2k3.append(math.sqrt(true_error_1 ** 2 + true_erorr_2 ** 2))
            # print(xk)
            print(y_next1, y_next2)

    res_h1_0, res_h1_1 = (y_next1, y_next2)  # y с чертой (-y)
    print(f'final estimation with h = {h1}, results = {xk, res_h1_0, res_h1_1}')

    xk = x0
    y_next1 = y1
    y_next2 = y2
    h2 = h / 2

    while xk + h2 < math.pi:
        (y_next1, y_next2) = step_k1k2k3(y_next1, y_next2, h2)
        xk += h2
        print('xk = ', xk)
        print(y_next1, y_next2)

        if xk + h2 >= math.pi:
            h_final = math.pi - xk
            (y_next1, y_next2) = step_k1k2k3(y_next1, y_next2, h_final)
            xk += h_final
            print('xk = ', xk)
            print(y_next1, y_next2)

    res_h2_0, res_h2_1 = (y_next1, y_next2)  # y с волной (~y)
    print(f'final estimation with h = {h2}, results = {xk, res_h2_0, res_h2_1}')
    # -y - h
    # ~y - h/2
    err1 = (res_h2_0 - res_h1_0) / (pow(2, s) - 1)  # ~Ri0
    err2 = (res_h2_1 - res_h1_1) / (pow(2, s) - 1)  # ~Ri1
    error = math.sqrt(err1 ** 2 + err2 ** 2)

    print(error)
    errors.append(error)

    h /= 2

print("Absolute error = ", y1_pi_true - res_h1_0, y2_pi_true - res_h1_1)
# print("Absolute error = ", y1_pi_true - y1_k, y2_pi_true - y2_k)


# print format(floatvalue, '.4f')

tol1 = 1e-5
tol2 = tol1 / pow(2, s)
tol3 = tol2 / pow(2, s + 1)
# tol3 < tol2 < tol1 = 10^(-5)

xk = x0
eps = 1e-04
s = 3

delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
h = pow(eps / delta, 1 / (s + 1)) / 2
h_i = h

h_list_k1k2k3 = []
points_k1k2k3 = []

print('eps = {:.6f}'.format(eps))
print('h = {:.6f}'.format(h))

print("Starting point y1_0, y2_0: {:.6f}, {:.6f}".format(y1, y2))
(y1_k, y2_k, y1_k_2, y2_k_2) = (y1, y2, y1, y2)

while xk + h_i < math.pi:
    y1_old = y1_k
    y2_old = y2_k
    (y1new, y2new) = step_k1k2k3(y1_k, y2_k, h_i)  # шаг с шагом h

    h_i2 = h_i / 2
    (y1newdva, y2newdva) = step_k1k2k3(y1_k, y2_k, h_i2)  # шаг с шагом h/2
    (y1_k_2, y2_k_2) = step_k1k2k3(y1newdva, y2newdva, h_i2)  # шаг с шагом h/2

    (y1_k, y2_k) = (y1new, y2new)  # новые найденные знач-я y1_k, y2_k

    err1 = (y1_k_2 - y1_k) / (1 - pow(2, -s))  # ~Ri0
    err2 = (y2_k_2 - y2_k) / (1 - pow(2, -s))  # ~Ri1
    error = math.sqrt(err1 ** 2 + err2 ** 2)

    if (error > eps * pow(2, s)):
        h_i /= 2
        print('h_i has been decreased by 2')
        y1_k = y1_old
        y2_k = y2_old
        y1_k_2 = y1_old
        y2_k_2 = y2_old

    elif (error > eps and error <= eps * pow(2, s)):
        print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
        points_k1k2k3.append(xk)
        h_list_k1k2k3.append(h_i)
        y1_k = y1_k_2
        y2_k = y2_k_2
        xk += h_i
        h_i /= 2
        # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
        # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')
        # y1real, y2real - трушные значения решения в точке Pi (из общего решения руками найти)

    elif (error >= eps / pow(2, s + 1) and error <= eps):
        print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
        points_k1k2k3.append(xk)
        h_list_k1k2k3.append(h_i)
        y1_k = y1new
        y2_k = y2new
        xk += h_i
        h_i = h_i
        # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
        # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')

    else:
        print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
        points_k1k2k3.append(xk)
        h_list_k1k2k3.append(h_i)
        y1_k = y1new
        y2_k = y2new
        xk += h_i
        h_i = h_i * 2

        # roo = pow((y1_k - y1real(x0)) * (y1_k - y1real(x0)) + (y2_k - y2real(x0)) * (y2_k - y2real(x0)), 0.5);
        # print(f'xk = {xk}, errors = {abs(roo), abs(error)}')

    print('y1_k = {:.8f}, y2_k = {:.8f}'.format(y1_k, y2_k))

# now final step to reach xk = pi
print('y1_k, y2_k before last step = ', y1_k, y2_k)
h_final = math.pi - xk
print('xk = ', xk)
print('h_final = ', h_final)
h_list_k1k2k3.append(h_final)
points_k1k2k3.append(math.pi)
(y1new, y2new) = step_k1k2k3(y1_k, y2_k, h_final)
(y1_k, y2_k) = (y1new, y2new)
print(f'final result at xk = {xk + h_final}: {y1_k, y2_k}')

print("Absolute error = ", y1_pi_true - y1_k, y2_pi_true - y2_k)

plt.title('k1, k2')
plt.plot(points, differences, c='blue', ls='--')
plt.scatter(points, differences, color='red')
plt.xlabel('$x$')
plt.ylabel('error')
plt.grid()

plt.title('k1, k2, k3')
plt.plot(points_k1k2k3, differences_k1k2k3, c='blue', ls='--')
plt.scatter(points_k1k2k3, differences_k1k2k3, color='red')
plt.xlabel('$x$')
plt.ylabel('error')
plt.grid()

plt.scatter(points, h_list, color='red', label='k1k2')
plt.plot(points, h_list, c='blue', ls='--')

plt.scatter(points_k1k2k3, h_list_k1k2k3, label='k1k2k3', c='orange')
plt.plot(points_k1k2k3, h_list_k1k2k3, c='purple', ls='--')

plt.xlabel('$x$')
plt.ylabel('$h_i$')
plt.grid()
plt.legend(loc='best')

eps_points = []
num_calculations_k1k2 = []
x0 = 0
s = 2

for i in range(1, 10):
    eps = math.pow(10, -i)

    # --- 3.3.2 -------------------------------------------------
    if eps == math.pow(10, -5):
        points_xk_332_k1k2 = []
        errors_ratio_332_k1k2 = []

        points_xk_332_k1k2.append(0)
        errors_ratio_332_k1k2.append(0)

        y1_prev = B * math.pi
        y2_prev = A * math.pi
        # -----------------------------------------------------------

    eps_points.append(eps)
    cnt = 0
    error = 1
    xk = x0
    delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
    h = pow(eps / delta, 1 / (s + 1))
    h_i = h

    # h_list_k1k2k3 = []
    # points_k1k2k3 = []

    print('eps = {:.6f}'.format(eps))
    print('h = {:.6f}'.format(h))

    print("Starting point y1_0, y2_0: {:.6f}, {:.6f}".format(y1, y2))
    (y1_k, y2_k, y1_k_2, y2_k_2) = (y1, y2, y1, y2)

    while xk + h_i < math.pi:
        y1_old = y1_k
        y2_old = y2_k
        (y1new, y2new) = step(y1_k, y2_k, h_i)  # шаг с шагом h

        h_i2 = h_i / 2
        (y1newdva, y2newdva) = step(y1_k, y2_k, h_i2)  # шаг с шагом h/2
        (y1_k_2, y2_k_2) = step(y1newdva, y2newdva, h_i2)  # шаг с шагом h/2
        cnt += 3

        (y1_k, y2_k) = (y1new, y2new)  # новые найденные знач-я y1_k, y2_k

        err1 = (y1_k_2 - y1_k) / (1 - pow(2, -s))  # ~Ri0
        err2 = (y2_k_2 - y2_k) / (1 - pow(2, -s))  # ~Ri1
        error = math.sqrt(err1 ** 2 + err2 ** 2)

        if (error > eps * pow(2, s)):
            h_i /= 2
            print('h_i has been decreased by 2')
            y1_k = y1_old
            y2_k = y2_old
            y1_k_2 = y1_old
            y2_k_2 = y2_old

        elif (error > eps and error <= eps * pow(2, s)):
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            y1_k = y1_k_2
            y2_k = y2_k_2
            xk += h_i
            h_i /= 2

        elif (error >= eps / pow(2, s + 1) and error <= eps):
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            y1_k = y1new
            y2_k = y2new
            xk += h_i
            h_i = h_i

        else:
            print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
            y1_k = y1new
            y2_k = y2new
            xk += h_i
            h_i = h_i * 2

        print('y1_k = {:.8f}, y2_k = {:.8f}'.format(y1_k, y2_k))

        # --- 3.3.2 -------------------------------------------------
        if eps == 10 ** (-5):
            c1, c2 = find_c1_c2(points_xk_332_k1k2[-1], y1_prev, y2_prev)
            points_xk_332_k1k2.append(xk)
            y1_t, y2_t = general_solution(points_xk_332_k1k2[-1], c1, c2)

            loc_err = sqrt((y1_k - y1_t) ** 2 + (y2_k - y2_t) ** 2)
            errors_ratio_332_k1k2.append(loc_err / error)

            y1_prev = y1_k
            y2_prev = y2_k
        # -----------------------------------------------------------
        # print('h before last step = ', h)

    # now final step to reach xk = pi
    print('y1_k, y2_k before last step = ', y1_k, y2_k)
    h_final = math.pi - xk
    print('xk = ', xk)
    print('h_final = ', h_final)
    (y1new, y2new) = step(y1_k, y2_k, h_final)
    (y1_k, y2_k) = (y1new, y2new)
    cnt += 3
    print(f'final result at xk = {xk + h_final}: {y1_k, y2_k}')
    # --------------- для 3.3.2 --------------------------------
    if eps == math.pow(10, -5):
        c1, c2 = find_c1_c2(points_xk_332_k1k2[-1], y1_prev, y2_prev)
        points_xk_332_k1k2.append(xk + h_final)
        y1_t, y2_t = general_solution(points_xk_332_k1k2[-1], c1, c2)

        loc_err = sqrt((y1_k - y1_t) ** 2 + (y2_k - y2_t) ** 2)
        errors_ratio_332_k1k2.append(loc_err / error)

        y1_prev = y1_k
        y2_prev = y2_k
    # ---------------------------------------------------------
    num_calculations_k1k2.append(cnt)
    print("Absolute error = ", y1_pi_true - y1_k, y2_pi_true - y2_k)
    print()

    eps_points = []
    num_calculations_k1k2k3 = []
    x0 = 0
    s = 3

    for i in range(1, 10):
        eps = math.pow(10, -i)

        # --- 3.3.2 -------------------------------------------------
        if eps == math.pow(10, -5):
            points_xk_332_k1k2k3 = []
            errors_ratio_332_k1k2k3 = []

            points_xk_332_k1k2k3.append(0)
            errors_ratio_332_k1k2k3.append(0)

            y1_prev = B * math.pi
            y2_prev = A * math.pi
            # -----------------------------------------------------------

        eps_points.append(eps)
        cnt = 0
        error = 1
        xk = x0
        delta = pow((1 / math.pi), s + 1) + pow(math.sqrt((A * A * math.pi) ** 2 + ((-B) * B * math.pi) ** 2), s + 1)
        h = pow(eps / delta, 1 / (s + 1))
        h_i = h

        # h_list_k1k2k3 = []
        # points_k1k2k3 = []

        print('eps = {:.6f}'.format(eps))
        print('h = {:.6f}'.format(h))

        print("Starting point y1_0, y2_0: {:.6f}, {:.6f}".format(y1, y2))
        (y1_k, y2_k, y1_k_2, y2_k_2) = (y1, y2, y1, y2)

        while xk + h_i < math.pi:
            y1_old = y1_k
            y2_old = y2_k
            (y1new, y2new) = step_k1k2k3(y1_k, y2_k, h_i)  # шаг с шагом h

            h_i2 = h_i / 2
            (y1newdva, y2newdva) = step_k1k2k3(y1_k, y2_k, h_i2)  # шаг с шагом h/2
            (y1_k_2, y2_k_2) = step_k1k2k3(y1newdva, y2newdva, h_i2)  # шаг с шагом h/2
            cnt += 3

            (y1_k, y2_k) = (y1new, y2new)  # новые найденные знач-я y1_k, y2_k

            err1 = (y1_k_2 - y1_k) / (1 - pow(2, -s))  # ~Ri0
            err2 = (y2_k_2 - y2_k) / (1 - pow(2, -s))  # ~Ri1
            error = math.sqrt(err1 ** 2 + err2 ** 2)

            if (error > eps * pow(2, s)):
                h_i /= 2
                print('h_i has been decreased by 2')
                y1_k = y1_old
                y2_k = y2_old
                y1_k_2 = y1_old
                y2_k_2 = y2_old

            elif (error > eps and error <= eps * pow(2, s)):
                print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
                y1_k = y1_k_2
                y2_k = y2_k_2
                xk += h_i
                h_i /= 2

            elif (error >= eps / pow(2, s + 1) and error <= eps):
                print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
                y1_k = y1new
                y2_k = y2new
                xk += h_i
                h_i = h_i

            else:
                print('xk = {:.6f}, h_i = {:.6f}'.format(xk, h_i), end=' ')
                y1_k = y1new
                y2_k = y2new
                xk += h_i
                h_i = h_i * 2

            print('y1_k = {:.8f}, y2_k = {:.8f}'.format(y1_k, y2_k))

            # --- 3.3.2 -------------------------------------------------
            if eps == 10 ** (-5):
                c1, c2 = find_c1_c2(points_xk_332_k1k2k3[-1], y1_prev, y2_prev)
                points_xk_332_k1k2k3.append(xk)
                y1_t, y2_t = general_solution(points_xk_332_k1k2k3[-1], c1, c2)

                loc_err = sqrt((y1_k - y1_t) ** 2 + (y2_k - y2_t) ** 2)
                errors_ratio_332_k1k2k3.append(loc_err / error)

                y1_prev = y1_k
                y2_prev = y2_k
            # -----------------------------------------------------------
            # print('h before last step = ', h)

        # now final step to reach xk = pi
        print('y1_k, y2_k before last step = ', y1_k, y2_k)
        h_final = math.pi - xk
        print('xk = ', xk)
        print('h_final = ', h_final)
        (y1new, y2new) = step_k1k2k3(y1_k, y2_k, h_final)
        (y1_k, y2_k) = (y1new, y2new)
        cnt += 3
        print(f'final result at xk = {xk + h_final}: {y1_k, y2_k}')
        # --------------- для 3.3.2 --------------------------------
        if eps == math.pow(10, -5):
            c1, c2 = find_c1_c2(points_xk_332_k1k2k3[-1], y1_prev, y2_prev)
            points_xk_332_k1k2k3.append(xk + h_final)
            y1_t, y2_t = general_solution(points_xk_332_k1k2k3[-1], c1, c2)

            loc_err = sqrt((y1_k - y1_t) ** 2 + (y2_k - y2_t) ** 2)
            errors_ratio_332_k1k2k3.append(loc_err / error)

            y1_prev = y1_k
            y2_prev = y2_k
        # ---------------------------------------------------------
        num_calculations_k1k2k3.append(cnt)
        print("Absolute error = ", y1_pi_true - y1_k, y2_pi_true - y2_k)
        print()

        plt.figure(figsize=(10, 5))
        plt.scatter(points_xk_332_k1k2, errors_ratio_332_k1k2, color='orange')
        plt.plot(points_xk_332_k1k2, errors_ratio_332_k1k2, c='purple', ls='--', label='k1,k2')

        plt.scatter(points_xk_332_k1k2k3, errors_ratio_332_k1k2k3, color='red')
        plt.plot(points_xk_332_k1k2k3, errors_ratio_332_k1k2k3, c='blue', ls='--', label='k1,k2,k3')

        plt.xlabel('$x$')
        plt.ylabel('error ratio')
        plt.grid()
        plt.legend(loc='best')

        print('eps list =', eps_points)
        print('k1,k2 num steps =', num_calculations_k1k2)
        print('k1,k2,k3 num steps =', num_calculations_k1k2k3)

        plt.figure(figsize=(10, 5))

        plt.scatter(eps_points[:-3], num_calculations_k1k2[:-3], color='orange')
        plt.plot(eps_points[:-3], num_calculations_k1k2[:-3], c='purple', ls='--', label='k1,k2')

        plt.scatter(eps_points[:-3], num_calculations_k1k2k3[:-3], color='red')
        plt.plot(eps_points[:-3], num_calculations_k1k2k3[:-3], c='blue', ls='--', label='k1,k2,k3')

        plt.xlabel('$\epsilon$')
        plt.ylabel('num calculations')
        plt.grid()
        plt.legend(loc='best')
