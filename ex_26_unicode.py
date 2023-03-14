import json
import pandas as pd
import os
import struct
# ----------------------------------------------------------------------------------------------------------------------
def from_symbol(str_u=u'\u2713'):

    bytes0 = str_u.encode("utf-8")
    bytes1 = bytes([226,156, 147])
    bytes2 = b'\xe2\x9c\x93'
    I = int.from_bytes(bytes0, byteorder='big', signed=False)
    bytes3 = bytes(struct.unpack('>4B', struct.pack('>L', I))[1:])
    str_u3 = bytes1.decode("utf-8")

    print(str_u)
    print(bytes0)
    print(bytes1)
    print(bytes2)
    print(bytes3)
    print(I)
    print(str_u3)

    return bytes0
# ----------------------------------------------------------------------------------------------------------------------
def int_from_bytes(bytes0):
    I = int.from_bytes(bytes0, byteorder='big', signed=False)
    str_u3 = bytes0.decode("utf-8")
    bytes1 = str_u3.encode("utf-8")

    print(bytes0)
    print(I)
    print(str_u3)
    print(bytes1)

    return str_u3
# ----------------------------------------------------------------------------------------------------------------------
def float_from_bytes(bytes0):

    bytes1 = bytes([b for b in bytes0])

    str_hex = bytes(bytes0).hex()
    str_hex = ''.join(format(b, '02x') for b in bytes0)

    #bb = [str_hex[i:i + 2] for i in range(0, len(str_hex), 2)]
    #bytes2 = b''.join([bytes.fromhex(b) for b in bb])
    bytes2 = b''.join([bytes.fromhex(str_hex[i:i + 2]) for i in range(0, len(str_hex), 2)])


    flt = struct.unpack('f', bytes2)[0]

    print(bytes0)
    print(bytes1)
    print(str_hex)
    print(flt)
    # print(str_u3)
    # print(bytes1)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_01():
    str_u = u'\u2713'
    from_symbol(str_u)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_02():
    bytes0 = bytes([226, 156, 147])
    int_from_bytes(bytes0)
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_03():
    value = 1.0
    bytes1 = bytearray(struct.pack('f', value))
    float_from_bytes(bytes1)

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex_04():

    res_char, res_I = [],[]
    for b1 in range(226,227):
        for b2 in range(0,255):
            for b3 in range(0, 255):
                try:
                    bytes0 = bytes([b1, b2, b3])
                    character = bytes0.decode("utf-8")
                    res_char.append(character)
                    res_I.append(int.from_bytes(bytes0, byteorder='big', signed=False))
                except:
                    pass

                if len(res_char)%80==0:
                    res_char.append('\n')
                    res_I.append(13)

    print(''.join(res_char))

    return

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex_02()
    ex_03()


