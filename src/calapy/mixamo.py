from selenium import webdriver
from selenium.webdriver.support import ui
from selenium.webdriver.common import keys
from selenium.common import exceptions
import time
import datetime
import os
import glob
import numpy as np
from . import strings as cp_strings
from . import clock as cp_clock
from . import download as cp_download
# from . import strings, clock, download


class MixamoBot:

    def __init__(self, directory_driver, directory_downloads=None, username=None, timeout=300):

        self.directory_driver = directory_driver

        if directory_downloads is None:
            self.directory_downloads = os.path.join(os.getcwd(), 'downloads')
        else:
            self.directory_downloads = directory_downloads

        if not (os.path.exists(self.directory_downloads)):
            os.makedirs(self.directory_downloads, exist_ok=True)

        if username is None:
            self.username = self.email = cp_strings.generate_a_random_email(
                range_1=[10, 15], range_2=[5, 10], end='.com')
        else:
            self.username = self.email = username

        self.timeout = timeout

        options = webdriver.ChromeOptions()
        options.add_experimental_option(
            "prefs", {"download.default_directory": self.directory_downloads,
                      "download.prompt_for_download": False})

        self.driver = webdriver.Chrome(executable_path=self.directory_driver, options=options)
        self.driver.set_page_load_timeout(self.timeout)
        self.driver.set_window_size(1000, 800)

        mixamo_website = 'https://www.mixamo.com/#/'
        self.driver.get(mixamo_website)

        # click Don't Enable
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('id', 'onetrust-reject-all-handler')
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "Don\'t Enable" button')
                else:
                    time.sleep(1)

        return None

    def signup(self, password=None, first_name=None, last_name=None, birthday=None):

        if password is None:
            password = cp_strings.generate_a_random_password(n=18)

        if first_name is None:
            first_name = cp_strings.generate_a_random_name([5, 10])

        if last_name is None:
            last_name = cp_strings.generate_a_random_name([5, 10])

        if birthday is None:
            birthday = dict(
                day='{}'.format(np.random.randint(low=1, high=28, size=1)[0].tolist()),
                month='{}'.format(np.random.randint(low=1, high=12, size=1)[0].tolist()),
                year='19{}'.format(np.random.randint(low=70, high=99, size=1)[0].tolist()))

        elif isinstance(birthday, dict):
            if birthday.get('day') is None:
                birthday['day'] = '1'
            if birthday.get('month') is None:
                birthday['month'] = '1'  # birthday['month'] = 1 is January, birthday['month'] = 2 is February ...
            if birthday.get('year') is None:
                birthday['year'] = '1990'
        else:
            raise ValueError('birthday has to be None or a dictionary')

        birthday = datetime.date(year=int(birthday['year']), month=int(birthday['month']), day=int(birthday['day']))

        # click SIGN UP
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('xpath', '//a[contains(., \'Sign up\')]').click()
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "Sign up" button')
                time.sleep(1)

        # type email
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('name', 'email').send_keys(self.username)
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "email" box')
                time.sleep(1)

        self.driver.find_element(
            'css selector', 'button.spectrum-Tool.spectrum-Tool--quiet.PasswordField-VisibilityToggle').click()

        self.driver.find_element('name', 'password').send_keys(password)

        time.sleep(1.0)
        self.driver.find_element(
            'xpath',
            '//button[@data-id=\'Signup-CreateAccountBtn\'][@name=\'submit\']'
            '[@class=\'spectrum-Button spectrum-Button--cta SpinnerButton SpinnerButton--right\']').click()

        # type name and bithday
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('name', 'firstname').send_keys(first_name)
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "firstname" box')
                time.sleep(1)

        self.driver.find_element('name', 'lastname').send_keys(last_name)

        self.driver.find_element('id', 'Signup-DateOfBirthChooser-Month').click()
        # time.sleep(2)

        name_month = birthday.__format__('%B')
        self.driver.find_element(
            'xpath', '//span[contains(., \'{name_month:s}\')]'.format(name_month=name_month)).click()

        # self.driver.find_element('name', 'day').click()
        # self.driver.find_element('name', 'day').send_keys(birthday.day)

        # self.driver.find_element('name', 'year').click()
        self.driver.find_element('name', 'bday-year').send_keys(birthday.year)

        time.sleep(1)
        self.driver.find_element(
            'xpath',
            '//button[@data-id=\'Signup-CreateAccountBtn\'][@name=\'submit\']'
            '[@class=\'spectrum-Button spectrum-Button--cta SpinnerButton SpinnerButton--right\']').click()

        # verify by email
        self.verify()

        self.clean()
        time.sleep(40)

        return None

    def login(self, password):

        # click LOG IN
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('xpath', '//a[contains(., \'Log in\')]').click()
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "Log in" button')
                time.sleep(1)

        # username
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('name', 'username').send_keys(self.username)
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "username" box')
                time.sleep(1)

        time.sleep(1.0)
        self.driver.find_element(
            'xpath',
            '//button[@data-id=\'EmailPage-ContinueButton\']'
            '[@class=\'spectrum-Button spectrum-Button--cta SpinnerButton SpinnerButton--right\']').click()

        # verify by email
        self.verify()

        # password
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('name', 'password').send_keys(password)
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "password" box')
                time.sleep(1)

        time.sleep(0.2)
        self.driver.find_element('xpath', '//button[@aria-label=\'Continue\']').click()

        self.clean()
        time.sleep(30)

        return None

    def clean(self):

        # click Don't Enable
        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                self.driver.find_element('id', 'onetrust-reject-all-handler').click()
                job_done = True
            except exceptions.NoSuchElementException:
                # add phone number
                try:
                    self.driver.find_element(
                        'xpath',
                        '//button[@data-id=\'PP-AddSecurityPhoneNumber-continue-btn\'][@type=\'submit\']'
                        '[@class=\'spectrum-Button spectrum-Button--cta\']').click()
                except exceptions.NoSuchElementException:
                    # add secondary email
                    try:
                        self.driver.find_element(
                            'xpath',
                            '//button[@data-id=\'PP-AddSecondaryEmail-continue-btn\'][@type=\'submit\']'
                            '[@class=\'spectrum-Button spectrum-Button--cta\']').click()
                    except exceptions.NoSuchElementException:
                        pass
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the "Don\'t Enable" button')
                else:
                    time.sleep(1)

        return None

    def verify(self, timeout=60):

        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                code_field = self.driver.find_element('xpath', '//input[@type=\'number\'][@data-id=\'CodeInput-0\']')
                job_done = True
            except exceptions.NoSuchElementException:
                try:
                    self.driver.find_element(
                        'xpath',
                        '//button[@data-id=\'Page-PrimaryButton\'][@name=\'submit\']'
                        '[@class=\'spectrum-Button spectrum-Button--cta SpinnerButton SpinnerButton--right '
                        'Page__spinner-btn\']').click()
                except exceptions.NoSuchElementException:
                    pass
                if timer.get_seconds() > timeout:
                    return None
                else:
                    time.sleep(1)

        typed_code = input(
            'A code was sent to "{email:s}".\n'
            'Manually enter the code here to verify your identiy.'.format(email=self.email))

        code_field.send_keys(typed_code)

        return None

    def activate_character(self, name_character, websites_search_actors):

        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                current_actor_element = self.driver.find_element('css selector', 'h2.text-center.h5')
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the name of the active character')
                time.sleep(1)
        current_actor_name = current_actor_element.text

        if name_character.upper() != current_actor_name.upper():
            self.driver.get(websites_search_actors)

            timer = cp_clock.Timer()
            job_done = False
            while not job_done:
                try:
                    self.driver.find_element('xpath', '//p[contains(., \'{}\')]'.format(name_character)).click()
                    job_done = True
                except exceptions.NoSuchElementException:
                    if timer.get_seconds() > self.timeout:
                        raise TimeoutError('I cannot find the character box')
                    time.sleep(1)

            timer = cp_clock.Timer()
            job_done = False
            while not job_done:
                try:
                    self.driver.find_element('xpath', '//button[contains(., \'Use this character\')]').click()
                    job_done = True
                except exceptions.NoSuchElementException:
                    if timer.get_seconds() > 20:
                        job_done = True
                    else:
                        time.sleep(1)
            # search_box.clear()
            time.sleep(30)

        return None

    def activate_animation(
            self, websites_animation, n_search_results_expected, i_search_result):

        self.driver.get(websites_animation)
        timer = cp_clock.Timer()
        elements_animations = self.driver.find_elements('css selector', 'div.product-overlay')
        n_animation_elements_from_web = len(elements_animations)
        while n_animation_elements_from_web != n_search_results_expected:

            if timer.get_seconds() > self.timeout:
                raise ValueError(
                    'The following condition is not met:\n'
                    'n_animation_elements_from_web = n_search_results_expected\n'
                    'n_animation_elements_from_web = {}\n'
                    'n_search_results_expected = {}\n'.format(
                        n_animation_elements_from_web, n_search_results_expected))
            time.sleep(1)
            elements_animations = self.driver.find_elements('css selector', 'div.product-overlay')
            n_animation_elements_from_web = len(elements_animations)

        elements_animations[i_search_result].click()

        return None

    def check_n_frames_default_from_web(self, n_frames_default_expected):

        n_frames_default_from_web = -1
        while n_frames_default_from_web == n_frames_default_expected:
            n_frames_default_from_web -= 1

        timer = cp_clock.Timer()

        while n_frames_default_from_web != n_frames_default_expected:

            if timer.get_seconds() > self.timeout:
                print(timer.get_seconds())
                raise ValueError(
                    'The following condition is not met:\n'
                    'n_frames_default_from_web = n_frames_default[i_animation]\n'
                    'n_frames_default_from_web = {}\n'
                    'n_frames_default_expected = {}'.format(n_frames_default_from_web, n_frames_default_expected))

            try:
                text_total_frames = self.driver.find_element('xpath', '//small[contains(., \'total frames\')]').text
            except exceptions.NoSuchElementException:
                time.sleep(1)
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot get the n_frames_default_from_web')
                else:
                    continue
            n_frames_default_from_web_str = ''
            for s in text_total_frames:
                try:
                    int(s)
                    n_frames_default_from_web_str += s
                except ValueError:
                    pass
            try:
                n_frames_default_from_web = int(n_frames_default_from_web_str)
            except ValueError:
                continue

        return None

    def get_n_frames_from_web(self, time_out=120):
        timer = cp_clock.Timer()
        n_frames_from_web_old = -2
        n_frames_from_web = -1
        while n_frames_from_web != n_frames_from_web_old:
            try:
                text_total_frames = self.driver.find_element('xpath', '//small[contains(., \'total frames\')]').text
                n_frames_from_web_old = n_frames_from_web
            except exceptions.NoSuchElementException:
                time.sleep(2)
                if timer.get_seconds() > time_out:
                    raise TimeoutError('I cannot get the n_frames_from_web')
                else:
                    continue
            n_frames_from_web_str = ''
            for s in text_total_frames:
                try:
                    int(s)
                    n_frames_from_web_str += s
                except ValueError:
                    pass
            try:
                n_frames_from_web = int(n_frames_from_web_str)
            except ValueError:
                continue

        return n_frames_from_web

    def set_animation_parameters(self, parameters, elements_variables_mixamo=None):
        if elements_variables_mixamo is None:
            elements_variables_mixamo = self.driver.find_elements(
                'css selector', 'input.input-text-unstyled.animation-slider-value.input-text-editable')[::-1]

        n_variables_mixamo = len(elements_variables_mixamo)
        for i_variable_mixamo in range(n_variables_mixamo):
            elements_variables_mixamo[i_variable_mixamo].send_keys(
                str(webdriver.common.keys.Keys.BACKSPACE * 5) +
                str(parameters[i_variable_mixamo]) + keys.Keys.ENTER)

        return None

    def set_animation_trim(self, start=0, end=100, elements_trim=None):

        if elements_trim is None:
            elements_trim = self.driver.find_elements('name', 'trim')

        elements_trim[0].send_keys(
            str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(start) + keys.Keys.ENTER)
        elements_trim[1].send_keys(
            str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(end) + keys.Keys.ENTER)

        return None

    def adjust_trim_to_make_n_frames_equal_or_larger_than_threshold(
            self, trims, trim_range, adjust_trim=None, threshold=60, elements_trim=None):

        if adjust_trim is None:
            adjust_trim = np.asarray([True, True], dtype=bool)

        if elements_trim is None:
            elements_trim = self.driver.find_elements('name', 'trim')

        n_frames_from_web = self.get_n_frames_from_web()
        frames_are_less_than_threshold = n_frames_from_web < threshold

        trim_start = trims[0]  # ???????????????
        adjust_trim_start = adjust_trim[0] and (
                trim_start > int(trim_range[0]))

        trim_end = trims[1]  # ???????????????
        adjust_trim_end = adjust_trim[1] and (
                trim_end < int(trim_range[1]))
        i = True
        adjust_trim_of_this_fbx = frames_are_less_than_threshold and (adjust_trim_start or adjust_trim_end)
        while adjust_trim_of_this_fbx:

            if adjust_trim_start and adjust_trim_end:
                if i:
                    trim_start -= 1
                    elements_trim[0].send_keys(
                        str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(trim_start) + keys.Keys.ENTER)
                    i = False
                else:
                    trim_end += 1
                    elements_trim[1].send_keys(
                        str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(trim_end) + keys.Keys.ENTER)
                    i = True

            elif adjust_trim_start:
                trim_start -= 1
                elements_trim[0].send_keys(
                    str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(trim_start) + keys.Keys.ENTER)

            elif adjust_trim_end:
                trim_end += 1
                elements_trim[1].send_keys(
                    str(webdriver.common.keys.Keys.BACKSPACE * 5) + str(trim_end) + keys.Keys.ENTER)

            n_frames_from_web = self.get_n_frames_from_web()
            frames_are_less_than_threshold = n_frames_from_web < threshold

            adjust_trim_start = adjust_trim[0] and (
                    trim_start > int(trim_range[0]))

            adjust_trim_end = adjust_trim[1] and (
                    trim_end < int(trim_range[1]))

            adjust_trim_of_this_fbx = frames_are_less_than_threshold and (
                    adjust_trim_start or adjust_trim_end)

        return None

    def download_animation(self, saved_fbx_as, rename_fbx_as):

        # self.driver.find_element('css selector', 'button.btn-block.btn.btn-primary').click()
        self.driver.find_element('xpath', '//button[contains(., \'Download\')]').click()

        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                menus = self.driver.find_elements('id', 'formControlsSelect')
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the downloading menus')
                time.sleep(1)

        element_menu_format = menus[0]
        menu_format = ui.Select(element_menu_format)
        menu_format.select_by_visible_text('FBX Binary(.fbx)')

        element_menu_skin = menus[1]
        menu_skin = ui.Select(element_menu_skin)
        menu_skin.select_by_visible_text('Without Skin')

        element_menu_frames_per_second = menus[2]
        menu_frames_per_second = ui.Select(element_menu_frames_per_second)
        menu_frames_per_second.select_by_visible_text('30')

        element_menu_keyframe_reduction = menus[3]
        menu_keyframe_reduction = ui.Select(element_menu_keyframe_reduction)
        menu_keyframe_reduction.select_by_visible_text('none')

        list_fbx_to_move = glob.glob(os.path.join(self.directory_downloads, '*.fbx'))
        n_fbx_to_move = len(list_fbx_to_move)
        if n_fbx_to_move != 0:
            raise Exception('n_fbx_to_move should be equal to 0 after the last download.\n'
                            'The value of n_fbx_to_move was {}.'.format(n_fbx_to_move))
        time.sleep(0.2)

        # download_elements = self.driver.find_elements('css selector', 'button.btn.btn-primary')
        # download_elements[1].click()
        self.driver.find_elements('xpath', "//button[contains(., 'Download')]")[1].click()

        cp_download.wait_downloading(saved_fbx_as, max_seconds_wait=self.timeout)

        list_fbx_to_move = glob.glob(os.path.join(self.directory_downloads, '*.fbx'))
        n_fbx_to_move = len(list_fbx_to_move)
        if n_fbx_to_move != 1:
            raise Exception('n_fbx_to_move should be equal to 1 after the last download.\n'
                            'The value of n_fbx_to_move was {}.'.format(n_fbx_to_move))
        os.rename(saved_fbx_as, rename_fbx_as)

        return None

    def download_t_pose(self, rename_fbx_as, timeout_dowloading=600):

        # self.driver.find_element('css selector', 'button.btn-block.btn.btn-primary').click()
        self.driver.find_element('xpath', '//button[contains(., \'Download\')]').click()

        timer = cp_clock.Timer()
        job_done = False
        while not job_done:
            try:
                menus = self.driver.find_elements('id', 'formControlsSelect')
                job_done = True
            except exceptions.NoSuchElementException:
                if timer.get_seconds() > self.timeout:
                    raise TimeoutError('I cannot find the downloading menus')
                time.sleep(1)

        element_menu_format = menus[0]
        menu_format = ui.Select(element_menu_format)
        menu_format.select_by_visible_text('FBX Binary(.fbx)')

        element_menu_pose = menus[1]
        menu_pose = ui.Select(element_menu_pose)
        menu_pose.select_by_visible_text('T-pose')

        list_fbx_to_rename = glob.glob(os.path.join(self.directory_downloads, '*.fbx'))
        n_fbx_to_rename = len(list_fbx_to_rename)
        if n_fbx_to_rename != 0:
            raise Exception('n_fbx_to_rename should be equal to 0 after the last download.\n'
                            'The value of n_fbx_to_rename was {}.'.format(n_fbx_to_rename))

        time.sleep(0.2)
        # download_elements = self.driver.find_elements('css selector', 'button.btn.btn-primary')
        # download_elements[1].click()
        self.driver.find_elements('xpath', "//button[contains(., 'Download')]")[1].click()

        timer = cp_clock.Timer()
        while n_fbx_to_rename != 1:
            if timer.get_seconds() > timeout_dowloading:
                raise TimeoutError('actor was not downloaded')
            list_fbx_to_rename = glob.glob(os.path.join(self.directory_downloads, '*.fbx'))
            n_fbx_to_rename = len(list_fbx_to_rename)
            time.sleep(1)

        os.rename(list_fbx_to_rename[0], rename_fbx_as)

        return None
