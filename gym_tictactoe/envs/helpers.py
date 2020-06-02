
def base_x_to_dec(base_list, x):
    base_list_rev = list(reversed(base_list))

    dec = 0
    for i in range(len(base_list_rev)):
        dec = dec + base_list_rev[i] * x ** i

    return dec


def dec_to_base_x(dec, x):
    base_list = []

    left = dec
    while left > 0:
        base_list.append(left % x)
        left = left // x

    return list(reversed(base_list))


def list_to_array(list, n):
    return [list[r::3] for r in range(3)]
