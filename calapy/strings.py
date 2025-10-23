
import inspect
import math
import typing
import numpy as np
import string
import random
from . import combinations as cp_combinations
from . import maths as cp_maths


def n_to_string_of_n_random_letters(n=1):
    # n is the number of random letters

    letters = string.ascii_letters
    string_of_n_random_letters = ''.join(random.choice(letters) for i in range(n))

    return string_of_n_random_letters


def n_to_string_of_n_random_uppercase_letters(n=1):
    # n is the number of random uppercase letters

    uppercase_letters = string.ascii_uppercase
    string_of_n_random_uppercase_letters = ''.join(random.choice(uppercase_letters) for i in range(n))

    return string_of_n_random_uppercase_letters


def n_to_string_of_n_random_lowercase_letters(n=1):
    # n is the number of random lowercase letters

    lowercase_letters = string.ascii_lowercase
    string_of_n_random_lowercase_letters = ''.join(random.choice(lowercase_letters) for i in range(n))

    return string_of_n_random_lowercase_letters


def n_to_string_of_n_random_numbers(n=1):
    # n is the number of random lowercase letters

    numbers = string.digits
    string_of_n_random_numbers = ''.join(random.choice(numbers) for i in range(n))

    return string_of_n_random_numbers


def n_to_string_of_n_random_digits(n=1):
    # n is the number of random digits

    letters_and_numbers = ''.join([string.ascii_letters, string.digits])
    string_of_n_random_digits = ''.join(random.choice(letters_and_numbers) for i in range(n))

    return string_of_n_random_digits


def generate_a_random_password(n=10):
    # n is the number

    if n > 3:

        m = math.ceil(n / 3)
        # diff = (m * 3) - n
        password = list(''.join([
            n_to_string_of_n_random_lowercase_letters(n=m),
            n_to_string_of_n_random_uppercase_letters(n=m),
            n_to_string_of_n_random_numbers(n=m)
        ]))

        random.shuffle(password)
        password = ''.join(password)

    else:
        raise ValueError('n')

    return password


def generate_a_random_email(range_1=None, range_2=None, end='.com'):
    # range_1 is the range of the random n of the email.
    # n is the number of the random digits before "@" of the email.
    # range_2 is the range of the random m of the email.
    # m is the number of the random digits between "@" and ".com" of the email.

    if range_1 is None:
        range_1 = [10, 15]

    if range_2 is None:
        range_2 = [5, 10]

    at = '@'
    len_first_letter = 1
    n_minus_1st_letter = random.randrange(range_1[0] - len_first_letter, range_1[1] - len_first_letter)

    m = random.randrange(range_2[0], range_2[1])

    email = ''.join([
        n_to_string_of_n_random_letters(n=len_first_letter),
        n_to_string_of_n_random_digits(n=n_minus_1st_letter), at,
        n_to_string_of_n_random_digits(n=m), end])

    return email


def generate_array_with_random_emails(shape_array=1, range_1=[10, 15], range_2=[5, 10], end='.com'):
    # shape_array is the shape of the array with random emails.
    # range_1 is the range of the random n of the e_th email.
    # n is the number of the random digits before "@" of the e_th email.
    # range_2 is the range of the random m of the e_th email.
    # m is the number of the random digits between "@" and ".com" of the e_th email.

    # letters_lowercase = string.ascii_lowercase
    # letters_uppercase = string.ascii_uppercase
    # numbers = string.digits
    # letters = ''.join([letters_lowercase, letters_lowercase])
    # letters_and_numbers = ''.join([string.ascii_letters, string.digits])

    at = '@'
    len_at = len(at)
    len_end = len(end)
    len_first_letter = 1

    try:
        n_axes_emails = len(shape_array)
        if not isinstance(shape_array, np.ndarray):
            shape_array = np.asarray(shape_array, dtype=int)
    except ValueError:
        shape_array = np.asarray([shape_array], dtype=int)
        n_axes_emails = 1

    n_minus_1st_letter = np.random.random_integers(
        low=(range_1[0] - len_first_letter), high=(range_1[1] - len_first_letter), size=shape_array)
    m = np.random.random_integers(low=(range_2[0]), high=(range_2[1]), size=shape_array)
    lens_emails_without_1st_letter_at_and_end = n_minus_1st_letter + m
    max_len_emails = lens_emails_without_1st_letter_at_and_end.max() + len_first_letter + len_at + len_end
    dtype = 'U' + str(max_len_emails)

    emails = np.empty(shape_array, dtype=dtype)

    for index_i in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_array):
        # print(index_i)
        emails[tuple(index_i)] = ''.join([
            n_to_string_of_n_random_letters(n=len_first_letter),
            n_to_string_of_n_random_digits(n=n_minus_1st_letter[tuple(index_i)]), at,
            n_to_string_of_n_random_digits(n=m[tuple(index_i)]), end])

    return emails


def generate_a_random_name(range_m=[5, 10]):
    # range_m is the range of the random m of the name.
    # m is the number of the random letters of the name.

    len_first_letter = 1
    m_minus_1st_letter = random.randrange(range_m[0] - len_first_letter, range_m[1] - len_first_letter)

    name = ''.join([
        n_to_string_of_n_random_uppercase_letters(n=len_first_letter),
        n_to_string_of_n_random_lowercase_letters(n=m_minus_1st_letter)])

    return name


def generate_array_with_random_names(shape_array=1, range_m=[5, 10]):
    # n is the number of random names
    # range_m is the range of the random m of the r_th name.
    # m is the number of the random letters of the r_th name.

    try:
        n_axes_names = len(shape_array)
        if not isinstance(shape_array, np.ndarray):
            shape_array = np.asarray(shape_array, dtype=int)
    except ValueError:
        shape_array = np.asarray([shape_array], dtype=int)
        n_axes_names = 1

    len_first_letter = 1
    m_minus_1st_letter = np.random.random_integers(
        low=(range_m[0] - len_first_letter), high=(range_m[1] - len_first_letter), size=shape_array)

    max_len_names = m_minus_1st_letter.max() + len_first_letter
    dtype = 'U' + str(max_len_names)

    names = np.empty(shape_array, dtype=dtype)

    for index_i in cp_combinations.n_conditions_to_combinations_on_the_fly(shape_array):
        # print(index_i)
        names[tuple(index_i)] = ''.join([
            n_to_string_of_n_random_uppercase_letters(n=len_first_letter),
            n_to_string_of_n_random_lowercase_letters(n=m_minus_1st_letter)])

    return names


def get_variable_name(variable, n_outer_frames=1):
    dict_variables = inspect.getouterframes(inspect.currentframe())[n_outer_frames].frame.f_locals
    keys, values = list(dict_variables.keys()), list(dict_variables.values())
    for i, v in enumerate(values):
        try:
            if v == variable:
                name = keys[i]
        except ValueError:
            if (v == variable).all():
                name = keys[i]
    return name


def get_list_of_n_variable_names(variables, n_outer_frames=1):

    n_variables = len(variables)
    list_of_names = [None] * n_variables  # type: list
    max_string_length = 0
    for i, v in enumerate(variables):
        list_of_names[i] = get_variable_name(v, n_outer_frames=n_outer_frames + 1)
        string_length_i = len(list_of_names[i])
        if string_length_i > max_string_length:
            max_string_length = string_length_i
    return np.asarray(list_of_names, dtype=('<U' + str(max_string_length)))


def format_float_to_str(value_float, n_decimals=None):
    if n_decimals is None:
        template = '{:f}'
    else:
        template = '{:.' + str(n_decimals) + 'f}'

    value_str = template.format(value_float)
    return value_str


def split_str_to_np(string, sep=None, maxsplit=-1, dtype=None, ignore=True, ignore_value=None, tolist=False):

    if ignore:
        if cp_maths.is_nan(ignore_value):
            elements = np.asarray(
                [e for e in string.split(sep=sep, maxsplit=maxsplit) if cp_maths.is_not_nan(e)],
                dtype=dtype)
        else:
            elements = np.asarray(
                [e for e in string.split(sep=sep, maxsplit=maxsplit) if e != ignore_value],
                dtype=dtype)
    else:
        elements = np.asarray(string.split(sep=sep, maxsplit=maxsplit), dtype=dtype)

    if tolist:
        return elements.tolist()
    else:
        return elements


def join(
        elements: typing.Union[None, int, float, str, list, tuple, np.ndarray],
        sep: typing.Optional[str] = None, frmt: typing.Optional[str] = None, ignore: bool = True, ignore_value=None):

    if isinstance(elements, np.ndarray):
        elements = elements.tolist()

    if frmt is None:
        sep = ' '

    if frmt is None:
        frmt = '{}'

    if isinstance(elements, (int, float, str)) or (elements is None):
        if ignore:
            if cp_maths.is_nan(ignore_value):
                if cp_maths.is_nan(elements):
                    return ''
                else:
                    return frmt.format(elements)
            else:
                if elements == ignore_value:
                    return ''
                else:
                    return frmt.format(elements)
        else:
            return frmt.format(elements)

    elif isinstance(elements, (list, tuple)):
        if ignore:
            if cp_maths.is_nan(ignore_value):
                return sep.join([frmt.format(e) for e in elements if cp_maths.is_not_nan(e)])
            else:
                return sep.join([frmt.format(e) for e in elements if e != ignore_value])
        else:
            return sep.join([frmt.format(e) for e in elements])
    else:
        raise TypeError('elements')
