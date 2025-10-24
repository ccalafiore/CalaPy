import numpy as np


def passwords_in_txt_files(directory_txt_file, n_digits=3):

    # characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890`¦!"$%^&*()_-+={}[]:;\'@#|\\,<.>/?'
    # print(characters)
    # characters = ''.join(sorted(characters))
    # print(characters)

    # characters = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}¦'
    characters = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                           'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                           's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

    print(len(characters))
    characters = np.asanyarray(sorted(characters), dtype='<U1')
    print(characters)
    print(len(characters))
    conditions_characters = np.empty(n_digits, dtype=object)
    for i_digit in range(n_digits):
        conditions_characters[i_digit] = characters

    passwords = make_combinations_of_conditions_as_unique_values(conditions_characters, type_conditions_values='<U1')

    np.savetxt(directory_txt_file, passwords, fmt="%s", delimiter="")


