import numpy as np
import sys
from ..maths import convert_to_int_or_float, rint
from ..array import FloatRange
from ..combinations import (
    conditions_to_combinations_on_the_fly as
    cp_conditions_to_combinations_on_the_fly)


def import_pygame():
    import pygame
    return pygame


pygame = import_pygame()


class Point:

    def __init__(self, point):

        if isinstance(point, Point):
            point_tmp = point.point
        elif isinstance(point, (np.ndarray, list, tuple)):
            point_tmp = point
        else:
            raise TypeError('point')

        # if isinstance(point_tmp, (np.ndarray, list, tuple)):
        x, y = point_tmp
        self.x = convert_to_int_or_float(x)
        self.y = convert_to_int_or_float(y)
        self.point = np.asarray([self.x, self.y])

    def convert_to_pygame_coordinates(self, size_frame):

        if isinstance(size_frame, (list, tuple)):
            half_size_frame = (np.asarray(size_frame) / 2)
        else:
            half_size_frame = (size_frame / 2)

        point_with_inverted_y = np.empty(2, dtype=self.point.dtype)
        point_with_inverted_y[0] = self.point[0]
        point_with_inverted_y[1] = -self.point[1]

        point_pg = PointPG((half_size_frame - .5) + point_with_inverted_y)

        return point_pg


class PointPG:

    def __init__(self, point, size_frame=None):

        if isinstance(point, PointPG):
            point_tmp = point.point
        elif isinstance(point, Point):
            point_tmp = point.convert_to_pygame_coordinates(size_frame).point
        elif isinstance(point, (np.ndarray, list, tuple)):
            point_tmp = point
        else:
            raise TypeError('point')

        # if isinstance(point_tmp, (np.ndarray, list, tuple)):
        x, y = point_tmp
        self.x = convert_to_int_or_float(x)
        self.y = convert_to_int_or_float(y)
        self.point = np.asarray([self.x, self.y])


class Line:

    def __init__(self, point_start, point_end):
        if isinstance(point_start, Point):
            self.point_start = point_start.point
        else:
            self.point_start = Point(point_start).point

        if isinstance(point_end, Point):
            self.point_end = point_end.point
        else:
            self.point_end = Point(point_end).point

    def convert_to_pygame_coordinates(self, size_frame):
        point_start_pg = Point(self.point_start).convert_to_pygame_coordinates(size_frame)
        point_end_pg = Point(self.point_end).convert_to_pygame_coordinates(size_frame)
        line_pg = LinePG(point_start_pg, point_end_pg)
        return line_pg


class LinePG:

    def __init__(self, point_start, point_end, size_frame=None):

        if isinstance(point_start, PointPG):
            self.point_start = point_start.point
        if isinstance(point_start, Point):
            self.point_start = point_start.convert_to_pygame_coordinates(size_frame).point
        else:
            self.point_start = PointPG(point_start).point

        if isinstance(point_end, PointPG):
            self.point_end = point_end.point
        if isinstance(point_end, Point):
            self.point_end = point_end.convert_to_pygame_coordinates(size_frame).point
        else:
            self.point_end = PointPG(point_end).point


class Rectangle:
    # (point, size, alignment_x=alignment_x, alignment_y=alignment_y)
    def __init__(self, point, size, alignment_x='c', alignment_y='c'):

        if isinstance(point, (list, tuple, np.ndarray)):
            point_tmp = Point(point)
        elif isinstance(point, Point):
            point_tmp = point
        elif isinstance(point, PointPG):
            raise TypeError('point')
        else:
            raise TypeError('point')

        self.width = self.w = convert_to_int_or_float(size[0])
        self.half_width = convert_to_int_or_float(self.w / 2)
        self.height = self.h = convert_to_int_or_float(size[1])
        self.half_height = convert_to_int_or_float(self.h / 2)

        alignment_x_low = alignment_x.lower()
        if alignment_x_low in ['c', 'center', 'centre']:
            self.centerx = point_tmp.x
            self.left = convert_to_int_or_float(self.centerx - (self.half_width - .5))
            self.right = convert_to_int_or_float(self.left + self.w)

        elif alignment_x_low in ['l', 'left']:
            self.left = point_tmp.x
            self.right = convert_to_int_or_float(self.left + self.w)
            self.centerx = convert_to_int_or_float(self.left + (self.half_width - .5))
        elif alignment_x_low in ['r', 'right']:
            self.right = point_tmp.x
            self.left = convert_to_int_or_float(self.right - self.w)
            self.centerx = convert_to_int_or_float(self.left + (self.half_width - .5))
        else:
            raise ValueError('alignment_x')

        alignment_y_low = alignment_y.lower()
        if alignment_y_low in ['c', 'center', 'centre']:
            self.centery = point_tmp.y
            self.bottom = convert_to_int_or_float(self.centery - (self.half_height - .5))
            self.top = convert_to_int_or_float(self.bottom + self.h)
        elif alignment_y_low in ['b', 'bottom']:
            self.bottom = point_tmp.y
            self.top = convert_to_int_or_float(self.bottom + self.h)
            self.centery = convert_to_int_or_float(self.bottom + (self.half_height - .5))
        elif alignment_y_low in ['t', 'top']:
            self.top = point_tmp.y
            self.bottom = convert_to_int_or_float(self.top - self.h)
            self.centery = convert_to_int_or_float(self.bottom + (self.half_height - .5))
        else:
            raise ValueError('alignment_y')

        self.bottomleft = np.asarray([self.left, self.bottom])
        self.bottomright = np.asarray([self.right, self.bottom])
        self.topleft = np.asarray([self.left, self.top])
        self.topright = np.asarray([self.right, self.top])
        self.center = np.asarray([self.centerx, self.centery])
        self.midleft = np.asarray([self.left, self.centery])
        self.midright = np.asarray([self.right, self.centery])
        self.midbottom = np.asarray([self.centerx, self.bottom])
        self.midtop = np.asarray([self.centerx, self.top])
        self.size = np.asarray([self.width, self.height])

    def convert_to_pygame_coordinates(self, size_frame):

        topleft_pg = Point(self.topleft).convert_to_pygame_coordinates(size_frame).point
        left_pg, top_pg = topleft_pg
        left_pg = rint(left_pg)
        top_pg = rint(top_pg)
        width = rint(self.width)
        height = rint(self.height)
        rect_pg = pygame.Rect(left_pg, top_pg, width, height)
        return rect_pg


class Cross:

    def __init__(self, center, size):

        if isinstance(center, Point):
            self.center = center.point
        else:
            self.center = Point(center).point

        if isinstance(size, (list, tuple)):
            len_size = len(size)
            if len_size == 1:
                self.size = np.asarray(size * 2)
            elif len_size == 2:
                self.size = np.asarray(size)
        elif isinstance(size, np.ndarray):
            len_shape = len(size.shape)
            if len_shape == 0:
                self.size = np.empty(2, dtype=size.dtype)
                self.size[:] = size
            elif len_shape == 1:
                if size.shape[0] == 1:
                    self.size = np.empty(2, dtype=size.dtype)
                    self.size[:] = size[0]
                elif size.shape[0] == 2:
                    self.size = size
        else:
            self.size = np.empty(2, dtype=type(size))
            self.size[:] = size

        self.half_size = self.size / 2

        # left, top = self.center - self.half_size
        # right, bottom = self.center + self.half_size
        left, top = self.center - (self.half_size - .5)
        right, bottom = self.center + (self.half_size - .5)

        self.left = convert_to_int_or_float(left)
        self.right = convert_to_int_or_float(right)
        self.top = convert_to_int_or_float(top)
        self.bottom = convert_to_int_or_float(bottom)

        self.point_start_horizontal_line = PointPG([self.left, self.center[1]])
        self.point_end_horizontal_line = PointPG([self.right, self.center[1]])

        self.point_start_vertical_line = Point([self.center[0], self.bottom])
        self.point_end_vertical_line = Point([self.center[0], self.top])

        self.horizontal_line = Line(self.point_start_horizontal_line, self.point_end_horizontal_line)
        self.vertical_line = Line(self.point_start_vertical_line, self.point_end_vertical_line)


class CrossPG:

    def __init__(self, center, size, size_frame):

        if isinstance(center, PointPG):
            self.center = center.point
        else:
            self.center = PointPG(center, size_frame=size_frame).point

        if isinstance(size, (list, tuple)):
            len_size = len(size)
            if len_size == 1:
                self.size = np.asarray(size * 2)
            elif len_size == 2:
                self.size = np.asarray(size)
        elif isinstance(size, np.ndarray):
            len_shape = len(size.shape)
            if len_shape == 0:
                self.size = np.empty(2, dtype=size.dtype)
                self.size[:] = size
            elif len_shape == 1:
                if size.shape[0] == 1:
                    self.size = np.empty(2, dtype=size.dtype)
                    self.size[:] = size[0]
                elif size.shape[0] == 2:
                    self.size = size
        else:
            self.size = np.empty(2, dtype=type(size))
            self.size[:] = size

        self.half_size = self.size / 2
        # left, top = self.center - self.half_size
        # right, bottom = self.center + self.half_size
        left, top = self.center - (self.half_size - .5)
        right, bottom = self.center + (self.half_size - .5)


        self.left = convert_to_int_or_float(left)
        self.right = convert_to_int_or_float(right)
        self.top = convert_to_int_or_float(top)
        self.bottom = convert_to_int_or_float(bottom)

        self.point_start_horizontal_line = PointPG([self.left, self.center[1]])
        self.point_end_horizontal_line = PointPG([self.right, self.center[1]])

        self.point_end_vertical_line = PointPG([self.center[0], self.top])
        self.point_start_vertical_line = PointPG([self.center[0], self.bottom])

        self.horizontal_line = LinePG(self.point_start_horizontal_line, self.point_end_horizontal_line)
        self.vertical_line = LinePG(self.point_start_vertical_line, self.point_end_vertical_line)


class Frame:

    def __init__(self, surface):
        self.surface = surface
        self.width, self.height = self.surface.get_size()
        self.w = self.width
        self.h = self.height
        self.half_width = self.half_w = convert_to_int_or_float(self.width / 2)
        self.half_height = self.half_h = convert_to_int_or_float(self.height / 2)
        self.size = np.asarray([self.w, self.h], dtype='i')
        self.half_size = np.asarray([self.half_w, self.half_h])

    def blit(self, source, point, alignment_x='c', alignment_y='c', area_sources=None, special_flags=0):

        # if isinstance(point, (list, tuple, np.ndarray)):
        #     point_pg = Point(point).convert_to_pygame_coordinates(self.size)
        # elif isinstance(point, Point):
        #     point_pg = point.convert_to_pygame_coordinates(self.size)
        # elif isinstance(point, PointPG):
        #     # point_pg = point
        #     raise TypeError('point')
        # else:
        #     raise TypeError('point')

        if isinstance(source, pygame.Surface):
            size_source = source.get_size()
        elif isinstance(source, Frame):
            size_source = source.size
        elif isinstance(source, StimText):
            size_source = source.frame.size
        else:
            raise TypeError('source')

        rect = Rectangle(point, size_source, alignment_x=alignment_x, alignment_y=alignment_y)
        rect_destination_pg = rect.convert_to_pygame_coordinates(self.size)

        # if isinstance(source, pygame.Surface):
        #     rect_destination_pg = source.get_rect()
        # elif isinstance(source, Frame):
        #     rect_destination_pg = source.surface.get_rect()
        # elif isinstance(source, StimText):
        #     rect_destination_pg = source.frame.surface.get_rect()
        # else:
        #     raise TypeError('source')


        # alignment_x_low = alignment_x.lower()
        # if alignment_x_low in ['c', 'center', 'centre']:
        #     rect_destination_pg.centerx = point_pg.x
        #     # rect_destination_pg.centerx = rint(point_pg.x + .5)
        # elif alignment_x_low in ['l', 'left']:
        #     rect_destination_pg.left = point_pg.x
        # elif alignment_x_low in ['r', 'right']:
        #     rect_destination_pg.right = point_pg.x
        # else:
        #     raise ValueError('alignment_x')
        #
        # alignment_y_low = alignment_y.lower()
        # if alignment_y_low in ['c', 'center', 'centre']:
        #     rect_destination_pg.centery = point_pg.y
        #     # rect_destination_pg.centery = rint(point_pg.y + .5)
        # elif alignment_y_low in ['b', 'bottom']:
        #     rect_destination_pg.bottom = point_pg.y
        # elif alignment_y_low in ['t', 'top']:
        #     rect_destination_pg.top = point_pg.y
        #     # rect_destination_pg.top = point_pg.y + 1
        # else:
        #     raise ValueError('alignment_y')

        if isinstance(source, pygame.Surface):
            self.surface.blit(
                source, rect_destination_pg,
                area=area_sources, special_flags=special_flags)
        elif isinstance(source, Frame):
            self.surface.blit(
                source.surface, rect_destination_pg,
                area=area_sources, special_flags=special_flags)
        elif isinstance(source, StimText):
            self.surface.blit(
                source.frame.surface, rect_destination_pg,
                area=area_sources, special_flags=special_flags)

    def blits(
            self, sources, point, alignment_x='c', alignment_y='c',
            area_sources=None, special_flags=0):

        n_sources = len(sources)
        for i in range(n_sources):
            self.blit(
                sources[i], point[i], alignment_x=alignment_x, alignment_y=alignment_y,
                area_sources=area_sources, special_flags=special_flags)

    def blits_in_different_columns_in_range_x(
            self, sources, range_x, point_y, alignment_x='c', alignment_y='c',
            area_sources=None, special_flags=0):

        n_sources = len(sources)
        myrange = FloatRange(range_x)
        alignment_x_low = alignment_x.lower()
        if alignment_x_low in ['c', 'center', 'centre']:
            points_x = myrange.centers_of_n_within_equal_ranges(n_sources)
        elif alignment_x_low in ['l', 'left']:
            points_x = myrange.starts_of_n_within_equal_ranges(n_sources)
        elif alignment_x_low in ['r', 'right']:
            points_x = myrange.stops_of_n_within_equal_ranges(n_sources)
        else:
            raise ValueError('alignment_x')

        for i in range(n_sources):
            self.blit(
                sources[i], [points_x[i], point_y], alignment_x=alignment_x, alignment_y=alignment_y,
                area_sources=area_sources, special_flags=special_flags)

    def blits_in_different_columns(
            self, sources, points_x, point_y, alignment_x='c', alignment_y='c',
            area_sources=None, special_flags=0):

        n_sources = len(sources)
        for i in range(n_sources):
            self.blit(
                sources[i], [points_x[i], point_y], alignment_x=alignment_x, alignment_y=alignment_y,
                area_sources=area_sources, special_flags=special_flags)

    def blits_in_different_rows(
            self, sources, point_x, points_y, alignment_x='c', alignment_y='c',
            area_sources=None, special_flags=0):

        n_sources = len(sources)
        for i in range(n_sources):
            self.blit(
                sources[i], [point_x, points_y[i]], alignment_x=alignment_x, alignment_y=alignment_y,
                area_sources=area_sources, special_flags=special_flags)

    def blits_in_different_columns_and_rows(
            self, sources, points_x, points_y, alignment_x='c', alignment_y='c',
            area_sources=None, special_flags=0):

        if not isinstance(points_x, (list, tuple, np.ndarray)):
            points_x = [points_x]
        if not isinstance(points_y, (list, tuple, np.ndarray)):
            points_y = [points_y]
        for (j, i), point_ji in cp_conditions_to_combinations_on_the_fly([points_x, points_y], order_outputs='iv'):
            self.blit(
                sources[j][i], point_ji, alignment_x=alignment_x, alignment_y=alignment_y,
                area_sources=area_sources, special_flags=special_flags)

    def fill(self, color, rect=None):
        if rect is None:
            rect_pg = None
        else:
            rect_pg = rect.convert_to_pygame_coordinates(self.size)

        self.surface.fill(color, rect_pg)
        return None

    def draw_line_from_points(self, point_start, point_end, color, width=1):

        if isinstance(point_start, Point):
            point_start_pg = point_start.convert_to_pygame_coordinates(self.size)
        elif isinstance(point_start, PointPG):
            point_start_pg = point_start
        else:
            point_start_pg = Point(point_start).convert_to_pygame_coordinates(self.size)

        if isinstance(point_end, Point):
            point_end_pg = point_end.convert_to_pygame_coordinates(self.size)
        elif isinstance(point_end, PointPG):
            point_end_pg = point_end
        else:
            point_end_pg = Point(point_end).convert_to_pygame_coordinates(self.size)

        line_pg = LinePG(point_start_pg, point_end_pg)
        pygame.draw.line(self.surface, color, line_pg.point_start, line_pg.point_end, width)

    def draw_line_from_line(self, line, color, width=1):
        if isinstance(line, Line):
            line_pg = line.convert_to_pygame_coordinates(self.size)
        elif isinstance(line, LinePG):
            line_pg = line
        else:
            raise TypeError('line')

        pygame.draw.line(self.surface, color, line_pg.point_start, line_pg.point_end, width)

    def draw_cross_from_point(self, point, size, color, width=1, alignment_x='c', alignment_y='c'):

        if isinstance(point, Point):
            point_pg = point.convert_to_pygame_coordinates(self.size)
        elif isinstance(point, PointPG):
            point_pg = point
        else:
            point_pg = Point(point).convert_to_pygame_coordinates(self.size)
        # TODO: cross alignment_x='c', alignment_y='c'
        cross_pg = CrossPG(point_pg, size, self.size)
        # draw horizontal line
        self.draw_line_from_line(cross_pg.horizontal_line, color, width=width)
        # draw horizontal line
        self.draw_line_from_line(cross_pg.vertical_line, color, width=width)

    def subsurface(self, rect):
        rect_pg = rect.convert_to_pygame_coordinates(self.size)
        my_subsurface = Frame(self.surface.subsurface(rect_pg))
        return my_subsurface

    def get_rect(self):
        left = -self.half_width
        bottom = -self.half_height
        rect = Rectangle([left, bottom], [self.width, self.height], alignment_x='l', alignment_y='b')
        return rect


def initiate_font():
    pygame.font.init()


class StimText:

    def __init__(
            self, text, size, color,
            alignment_rows='c', name=None, bold=False, italic=False, distance_rows=1,
            antialias=False, background=None):

        if isinstance(text, str):
            self.text = text
        elif isinstance(text, (list, tuple, np.ndarray)):
            self.text = '\n'.join(text)
        else:
            raise TypeError('text')

        self.rows = self.text.splitlines()
        self.n_rows = self.R = len(self.rows)

        self.size = size
        self.color = color
        self.alignment_rows = alignment_rows
        self.name = name
        self.bold = bold
        self.italic = italic
        self.distance_rows = distance_rows
        self.antialias = antialias
        self.background = background

        font = pygame.font.SysFont(name, size, bold=bold, italic=italic)
        if self.R == 1:
            self.frame = Frame(font.render(self.rows[0], self.antialias, self.color, self.background))
        else:
            frames: list = [None] * self.R
            widths_lines = np.empty(self.R, dtype='i')
            heights_lines = np.empty(self.R, dtype='i')
            for r in range(self.R):
                frames[r] = Frame(font.render(self.rows[r], self.antialias, self.color, self.background))
                widths_lines[r], heights_lines[r] = frames[r].size

            max_width_lines = int(widths_lines.max(initial=0))
            max_height_lines = int(heights_lines.max(initial=0))
            height = ((max_height_lines * self.distance_rows) * (self.R - 1)) + max_height_lines
            self.frame = Frame(pygame.Surface([max_width_lines, height]))

            if self.background is None:
                diff_color = 50
                threshould = 255 - diff_color
                background_tmp = [0, 0, 0]
                for c in range(3):
                    if self.color[c] <= threshould:
                        background_tmp[c] = self.color[c] + diff_color
                    elif self.color[c] > threshould:
                        background_tmp[c] = self.color[c] - diff_color

                # background_tmp = [self.color[c] for c in range(3)]
                # if 0 <= background_tmp[0] < 255:
                #     background_tmp[0] += 1
                # elif background_tmp[0] == 255:
                #     background_tmp[0] -= 1
                # else:
                #     raise ValueError('color[0]')
                # background_tmp = [50, 200, 50]
                # if 255 in self.color:
                #     background_tmp = [0, 0, 0]
                # else:
                #     background_tmp = [255, 255, 255]

                self.frame.fill(background_tmp)
                # self.frame.fill(self.color)
                self.frame.surface.set_colorkey(background_tmp)
            else:
                self.frame.surface.fill(self.background)

            left_first_line = -self.frame.half_width
            bottom_first_line = self.frame.half_height - max_height_lines
            rect_first_line = Rectangle(
                [left_first_line, bottom_first_line],
                [self.frame.width, max_height_lines],
                alignment_x='l', alignment_y='b')
            surface_first_line = self.frame.subsurface(rect_first_line)

            bottom_other_lines = -self.frame.half_height
            height_other_lines = self.frame.height - max_height_lines
            rect_other_lines = Rectangle(
                [left_first_line, bottom_other_lines],
                [self.frame.width, height_other_lines],
                alignment_x='l', alignment_y='b')
            surface_other_lines = self.frame.subsurface(rect_other_lines)

            alignment_rows_lower = alignment_rows.lower()
            if alignment_rows_lower in ['c', 'center', 'centre']:
                x_lines = 0
            elif alignment_rows_lower in ['l', 'left']:
                x_lines = -max_width_lines / 2
            elif alignment_rows_lower in ['r', 'right']:
                x_lines = max_width_lines / 2
            else:
                raise ValueError('alignment_x')

            surface_first_line.blit(
                frames[0], [x_lines, surface_first_line.half_height], alignment_x=alignment_rows, alignment_y='t')

            num_sources = np.arange(self.R - 1)
            y_lines = ((num_sources[::-1] - num_sources.mean() - 0.5) * (max_height_lines * self.distance_rows))

            surface_other_lines.blits_in_different_rows(
                frames[1:], x_lines, y_lines, alignment_x=alignment_rows, alignment_y='b')

    # def draw(self, destination, point, area_sources=None, special_flags=0):
    #
    #     if isinstance(point, PointPG):
    #         raise TypeError('point')
    #
    #     num_sources = np.arange(self.R)
    #     points_y = point[1] + ((num_sources[::-1] - num_sources.mean()) * self.size)
    #     point_x = point[0]
    #
    #     if isinstance(destination, pygame.Surface):
    #         Frame(destination).blits_in_different_rows(
    #             self.frames, point_x, points_y,
    #             area_sources=area_sources, special_flags=special_flags)
    #     elif isinstance(destination, Frame):
    #         destination.blits_in_different_rows(
    #             self.frames, point_x, points_y,
    #             area_sources=area_sources, special_flags=special_flags)


def list_of_text_to_list_of_stimtext(
        text, size, same_color=True, color=None,
        alignment_rows='c', name=None, bold=False, italic=False, distance_rows=1,
        antialias=False, background=None):

    if color is None:
        color = [255, 255, 255]
        same_color = True

    n_texts = len(text)
    frames: list = [None] * n_texts

    if same_color:
        for i in range(n_texts):
            frames[i] = StimText(
                text[i], size, color, alignment_rows=alignment_rows,
                name=name, bold=bold, italic=italic,
                distance_rows=distance_rows, antialias=antialias, background=background)
    else:
        for i in range(n_texts):
            frames[i] = StimText(
                text[i], size, color[i], alignment_rows=alignment_rows,
                name=name, bold=bold, italic=italic,
                distance_rows=distance_rows, antialias=antialias, background=background)

    return frames


def surfaces_to_sizes(surfaces):
    n_surfaces = len(surfaces)
    sizes: list = [None] * n_surfaces
    for i in range(n_surfaces):
        sizes[i] = surfaces[i].get_size()
    return sizes


def print_names_fonts():
    names = pygame.font.get_fonts()
    print('There are {n} fonts:'.format(n=len(names)))
    for name_i in names:
        print(name_i)


def font_names():
    names = pygame.font.get_fonts()
    return names


def update(rect=None):
    if rect is None:
        pygame.display.flip()
    else:
        info = pygame.display.Info()
        size_screen = [info.current_w, info.current_h]
        rect_pg = rect.convert_to_pygame_coordinates(size_screen)
        pygame.display.update(rect_pg)
    return None


def flip_until_key_is_pressed(key):
    wait = True
    while wait:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == key:
                    wait = False

        pygame.display.flip()
