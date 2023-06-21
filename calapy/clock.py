

import time as tm
import datetime
import math
from . import maths


hours_per_day = 24
minutes_per_hour = 60
minutes_per_day = hours_per_day * minutes_per_hour
seconds_per_minute = 60
seconds_per_hour = minutes_per_hour * seconds_per_minute
seconds_per_day = hours_per_day * seconds_per_hour
milliseconds_per_second = 1000
milliseconds_per_minute = milliseconds_per_second * seconds_per_minute
milliseconds_per_hour = milliseconds_per_minute * minutes_per_hour
milliseconds_per_day = milliseconds_per_hour * hours_per_day
microseconds_per_millisecond = 1000
microseconds_per_second = microseconds_per_millisecond * milliseconds_per_second
microseconds_per_minute = microseconds_per_second * seconds_per_minute
microseconds_per_hour = microseconds_per_minute * minutes_per_hour
microseconds_per_day = microseconds_per_hour * hours_per_day


class SecTimer:

    def __init__(self):
        self.end_second = None
        self.delta_seconds = None
        self.start_second = tm.time()

    def _define_delta_seconds(self):
        self.end_second = tm.time()
        self.delta_seconds = self.end_second - self.start_second

    def get_seconds(self):
        self._define_delta_seconds()
        return self.delta_seconds

    def get_milliseconds(self):
        self._define_delta_seconds()
        milliseconds = self.delta_seconds * milliseconds_per_second
        return milliseconds


class Timer:

    def __init__(self):

        self.end_datetime = None
        self.delta_time_total = None

        self.current_tick = None
        self.delta_time_ticks = None

        self.previous_tick = self.start_datetime = datetime.datetime.today()

    def _define_delta_time(self, start_datetime, end_datetime):

        timedelta_builtin = end_datetime - start_datetime
        delta_time = TimeDelta(
            days=timedelta_builtin.days,
            seconds=timedelta_builtin.seconds,
            microseconds=timedelta_builtin.microseconds)

        return delta_time

    def _define_delta_time_total(self):

        self.end_datetime = datetime.datetime.today()
        self.delta_time_total = self._define_delta_time(
            start_datetime=self.start_datetime, end_datetime=self.end_datetime)

    def _define_delta_time_ticks(self):
        self.current_tick = datetime.datetime.today()
        self.delta_time_ticks = self._define_delta_time(
            start_datetime=self.previous_tick, end_datetime=self.current_tick)

    def _tick(self):
        self._define_delta_time_ticks()
        self.previous_tick , self.current_tick = self.current_tick, None
        return self.delta_time_ticks

    def get_delta_time_total(self):
        self._define_delta_time_total()
        return self.delta_time_total

    def get_delta_time_ticks(self):
        self._define_delta_time_ticks()
        return self.delta_time_ticks

    def get_seconds_total(self):
        self._define_delta_time_total()
        seconds = self.delta_time_total.to_seconds()
        return seconds

    def get_milliseconds_total(self):
        self._define_delta_time_total()
        milliseconds = self.delta_time_total.to_milliseconds()
        return milliseconds

    def get_seconds_ticks(self):
        self._define_delta_time_ticks()
        seconds = self.delta_time_ticks.to_seconds()
        return seconds

    def get_milliseconds_ticks(self):
        self._define_delta_time_ticks()
        milliseconds = self.delta_time_ticks.to_milliseconds()
        return milliseconds
    
    def wait(self, seconds):

        correction_secs = -0.0005

        passed_seconds = self.get_seconds_ticks()
        remaining_seconds = seconds - passed_seconds + correction_secs
        if remaining_seconds > 0:
            tm.sleep(remaining_seconds)

        # passed_seconds = self.get_seconds_ticks()
        # remaining_seconds = seconds - passed_seconds
        # while remaining_seconds > 0:
        #     passed_seconds = self.get_seconds_ticks()
        #     remaining_seconds = seconds - passed_seconds

        self._tick()

        return self.delta_time_ticks.to_seconds()


class TimeDelta:
    def __init__(self, days=0, hours=0, minutes=0, seconds=0, milliseconds=0, microseconds=0):
        self.type = 'day'

        self.microseconds = maths.convert_to_int_or_float(
            (maths.convert_to_int_or_float(days) * microseconds_per_day) +
            (maths.convert_to_int_or_float(hours) * microseconds_per_hour) +
            (maths.convert_to_int_or_float(minutes) * microseconds_per_minute) +
            (maths.convert_to_int_or_float(seconds) * microseconds_per_second) +
            (maths.convert_to_int_or_float(milliseconds) * microseconds_per_millisecond) +
            maths.convert_to_int_or_float(microseconds))

        self.milliseconds = math.floor(self.microseconds / microseconds_per_millisecond)
        self.microseconds -= (self.milliseconds * microseconds_per_millisecond)

        self.seconds = math.floor(self.milliseconds / milliseconds_per_second)
        self.milliseconds -= (self.seconds * milliseconds_per_second)

        self.minutes = math.floor(self.seconds / seconds_per_minute)
        self.seconds -= (self.minutes * seconds_per_minute)

        self.hours = math.floor(self.minutes / minutes_per_hour)
        self.minutes -= (self.hours * minutes_per_hour)

        self.days = math.floor(self.hours / hours_per_day)
        self.hours -= (self.days * hours_per_day)

    def to_seconds(self):

        seconds = (
            (self.days * seconds_per_day) +
            (self.hours * seconds_per_hour) +
            (self.minutes * seconds_per_minute) +
            self.seconds +
            maths.convert_to_int_or_float(self.milliseconds / milliseconds_per_second) +
            maths.convert_to_int_or_float(self.microseconds / microseconds_per_second))

        return seconds

    def to_milliseconds(self):

        milliseconds = (
            (self.days * milliseconds_per_day) +
            (self.hours * milliseconds_per_hour) +
            (self.minutes * milliseconds_per_minute) +
            (self.seconds * milliseconds_per_second) +
            self.milliseconds +
            maths.convert_to_int_or_float(self.microseconds / microseconds_per_second))

        return milliseconds
