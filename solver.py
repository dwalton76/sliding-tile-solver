#!/usr/bin/env python3

from collections import deque
from copy import copy
from pprint import pformat
from subprocess import check_output
import argparse
import datetime as dt
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)

class ImplementThis(Exception):
    pass

class NoSteps(Exception):
    pass

class NoIDASolution(Exception):
    pass

class IllegalMove(Exception):
    pass

class PuzzleError(Exception):
    pass


def is_square(integer):
    root = math.sqrt(integer)

    if int(root + 0.5) ** 2 == integer:
        return True
    else:
        return False


def pretty_time(delta):
    delta = str(delta)

    if delta.startswith('0:00:00.'):
        delta_us = int(delta.split('.')[1])
        delta_ms = int(delta_us/1000)

        if delta_ms >= 500:
            return "\033[91m%sms\033[0m" % delta_ms
        else:
            return "%sms" % delta_ms

    elif delta.startswith('0:00:01.'):
        delta_us = int(delta.split('.')[1])
        delta_ms = 1000 + int(delta_us/1000)
        return "\033[91m%sms\033[0m" % delta_ms

    else:
        return "\033[91m%s\033[0m" % delta


def reverse_moves(moves):
    result = []

    for move in reversed(moves):
        if move == 'u':
            result.append('d')
        elif move == 'd':
            result.append('u')
        elif move == 'l':
            result.append('r')
        elif move == 'r':
            result.append('l')

    return result


class LookupTable(object):

    def __init__(self, parent, filename, state_type, state_target, linecount, size, max_depth=None):
        self.parent = parent
        self.filename = filename
        self.filename_gz = filename + '.gz'
        self.desc = filename.replace('lookup-table-', 'tilesize-').replace('.txt', '')
        self.filename_exists = False
        self.linecount = linecount
        self.max_depth = max_depth
        self.size = size

        assert self.filename.startswith('lookup-table'), "We only support lookup-table*.txt files"
        assert self.filename.endswith('.txt'), "We only support lookup-table*.txt files"
        assert self.linecount, "%s linecount is %s" % (self, self.linecount)

        if not os.path.exists(self.filename):

            # If we do not have the gzip file download the various parts from
            # the sliding-tile-solver-lookup-tables and then cat them together.
            # We have to jump through these hoops because github does not allow
            # files over 100M.
            if not os.path.exists(self.filename_gz):
                if self.filename_gz in ('lookup-table-16.txt.gz',
                                        'lookup-table-16-x-1-6.txt.gz',
                                        'lookup-table-16-x-7-12.txt.gz'):
                    for part in ('aa', 'ab', 'ac', 'ad', 'ae'):
                        url = "https://github.com/dwalton76/sliding-tile-solver-lookup-tables/raw/master/%s.part-%s" % (self.filename_gz, part)
                        log.info("Downloading table via 'wget %s'" % url)
                        subprocess.call(['wget', url])

                    subprocess.call('cat %s.part-* > %s' % (self.filename_gz, self.filename_gz), shell=True)

                    for part in ('aa', 'ab', 'ac', 'ad', 'ae'):
                        os.unlink("%s.part-%s" % (self.filename_gz, part))

                elif self.filename_gz == 'lookup-table-16-x-13-15.txt.gz':
                    url = "https://github.com/dwalton76/sliding-tile-solver-lookup-tables/raw/master/%s" % self.filename_gz
                    log.info("Downloading table via 'wget %s'" % url)
                    subprocess.call(['wget', url])

            log.warning("gunzip %s" % self.filename_gz)
            subprocess.call(['gunzip', self.filename_gz])

        # Find the state_width for the entries in our .txt file
        with open(self.filename, 'r') as fh:
            first_line = next(fh)
            self.line_width = len(first_line)
            (state, steps) = first_line.split(':')
            self.state_width = len(state)

        self.filename_exists = True
        self.state_type = state_type
        self.state_target = state_target
        self.cache = {}
        self.fh_txt = open(self.filename, 'r')

    def __str__(self):
        return self.desc

    def state(self):

        if self.state_type == 'all':
            return ''.join(["%x" % x for x in self.parent.state[1:]])

        elif self.state_type == '1-6':
            return ''.join(["%x" % number if 0 <= number <= 6 else 'x' for number in self.parent.state[1:]])

        elif self.state_type == '7-12':
            return ''.join(["%x" % number if (7 <= number <= 12 or number == 0) else 'x' for number in self.parent.state[1:]])

        elif self.state_type == '13-15':
            return ''.join(["%x" % number if (13 <= number <= 15 or number == 0) else 'x' for number in self.parent.state[1:]])

        elif self.state_type == '1-2-3':
            return ''.join(["%x" % number if 0 <= number <= 3 else 'x' for number in self.parent.state[1:]])

        elif self.state_type == '4-5-6':
            return ''.join(["%x" % number if 0 <= number <= 6 else 'x' for number in self.parent.state[1:]])

        elif self.state_type == '11-16-21':
            foo = []
            for number in self.parent.state[1:]:
                if 0 <= number <= 6 or number == 11:
                    foo.append("%x" % number)
                elif number == 16:
                    foo.append('g')
                elif number == 21:
                    foo.append('l')
                else:
                    foo.append('x')
            return ''.join(foo)

        elif self.state_type == '7-13-19':
            foo = []
            for number in self.parent.state[1:]:
                if 0 <= number <= 7 or number == 13:
                    foo.append("%x" % number)
                elif number == 19:
                    foo.append('j')
                else:
                    foo.append('x')
            return ''.join(foo)

        elif self.state_type == '25-31':
            foo = []
            for number in self.parent.state[1:]:
                if 0 <= number <= 7 or number == 13:
                    foo.append("%x" % number)
                elif number == 19:
                    foo.append('j')
                elif number == 25:
                    foo.append('p')
                elif number == 31:
                    foo.append('v')
                else:
                    foo.append('x')
            return ''.join(foo)

        else:
            raise ImplementThis("support state_type %s" % self.state_type)

    def binary_search(self, state_to_find):
        first = 0
        last = self.linecount - 1
        #log.info("")
        #log.info("")
        #log.info("%s: line_width %d, state_width %d, state_to_find %s" % (self, self.line_width, self.state_width, pformat(state_to_find)))

        while first <= last:
            midpoint = int((first + last)/2)
            self.fh_txt.seek(midpoint * self.line_width)
            line = self.fh_txt.readline().rstrip()

            try:
                (state, steps) = line.split(':')
            except Exception:
                log.warning("%s: midpoint %d, line_width %d, state_to_find %s, line %s" % (self, midpoint, self.line_width, state_to_find, line))
                raise

            if state == state_to_find:
                return line
            else:
                if state_to_find < state:
                    #log.info("%s: LEFT , first %d, midpoint %d, last %d, state %s" % (self, first, midpoint, last, pformat(state)))
                    last = midpoint-1
                else:
                    #log.info("%s: RIGHT, first %d, midpoint %d, last %d, state %s" % (self, first, midpoint, last, pformat(state)))
                    first = midpoint+1

        return None

    def steps(self, state_to_find=None):
        """
        Return a list of the steps found in the lookup table for the current puzzle state
        """
        if state_to_find is None:
            state_to_find = self.state()

        try:
            return self.cache[state_to_find]
        except KeyError:
            line = self.binary_search(state_to_find)

            if not line:
                self.cache[state_to_find] = None
                return None

            try:
                (state, steps) = line.split(':')
            except Exception:
                log.info("%s: %s, steps %s" % (self, line, steps))
                raise

            if state == state_to_find:

                if steps.isdigit():
                    self.cache[state_to_find] = int(steps)
                else:
                    self.cache[state_to_find] = list(steps)

                return self.cache[state_to_find]

            self.cache[state_to_find] = None
            return None

    def steps_length(self, state=None):
        return len(self.steps(state))

    def solve(self):

        if not self.filename_exists:
            raise PuzzleError("%s does not exist" % self.filename)

        while True:
            state = self.state()

            if self.state_target == 'TBD':
                log.info("%s: solve() state %s vs state_target %s" % (self, state, pformat(self.state_target)))

            if state == self.state_target:
                break

            steps = self.steps(state)

            if steps is None:
                raise NoSteps("%s: state %s does not have steps" % (self, state))

            if not steps:
                return

            for step in steps:
                self.parent.move_xyz(step)


class LookupTableCumulative(LookupTable):
    pass


class LookupTableIDA(LookupTable):

    def __init__(self, parent, filename, state_type, state_target, prune_tables, linecount, size):
        LookupTable.__init__(self, parent, filename, state_type, state_target, linecount, size)
        self.prune_tables = prune_tables

    def ida_heuristic(self):
        cost_to_goal = 0
        cumulative_tables_cost = 0
        debug = False

        for pt in self.prune_tables:

            if isinstance(pt, LookupTable):
                pt_state = pt.state()
                pt_steps = pt.steps(pt_state)

                if pt_state == pt.state_target:
                    len_pt_steps = 0

                    if debug:
                        log.info("%s: pt_state %s, cost 0, at target" % (pt, pt_state))

                elif pt_steps is not None:
                    if isinstance(pt_steps, list):

                        if pt_steps:
                            len_pt_steps = int(pt_steps[0])
                        else:
                            len_pt_steps = 0

                    elif isinstance(pt_steps, int):
                        len_pt_steps = pt_steps

                    else:
                        raise Exception("We should not be here")

                    if debug:
                        log.info("%s: pt_state %s, cost %d, pt_steps %s" % (pt, pt_state, len_pt_steps, pformat(pt_steps)))

                else:
                    raise PuzzleError("%s does not have steps for %s  state_width %d" % (pt, pt_state, pt.state_width))
            else:
                len_pt_steps = pt.cost()

            if isinstance(pt, LookupTableCumulative):
                cumulative_tables_cost += len_pt_steps
            else:
                # Use the max cost among all prune tables
                if len_pt_steps > cost_to_goal:
                    cost_to_goal = len_pt_steps

        # The costs from LookupTableCumulative prune tables can be totaled
        if cumulative_tables_cost > cost_to_goal:
            cost_to_goal = cumulative_tables_cost

        return cost_to_goal

    def ida_search(self, steps_to_here, threshold, prev_step,
                  prev_state, prev_moves_to_here, prev_empty_index, prev_empty_type):
        """
        https://algorithmsinsight.wordpress.com/graph-theory-2/ida-star-algorithm-in-general/
        """
        debug = False
        cost_to_here = len(steps_to_here)
        cost_to_goal = self.ida_heuristic()
        f_cost = cost_to_here + cost_to_goal

        assert steps_to_here == self.parent.moves_to_here,\
            "cost_to_here %d %s, moves_to_here %d %s" %\
            (cost_to_here, pformat(steps_to_here), len(self.parent.moves_to_here), pformat(self.parent.moves_to_here))

        # This looks a little odd because the puzzle may be in a state where we
        # find a hit in our lookup table and we could execute the steps
        # per the table and be done with our IDA search.
        #
        # That could cause us to return a longer moves_to_here but with the benefit
        # of the searching being faster....I am torn on whether to return False
        # here or not.
        if f_cost > threshold:
            if debug:
                log.info("%s: PRUNE cost_to_here %d %s, cost_to_goal %d, f_cost %d, threshold %d, moves_to_here %s" %\
                         (self, cost_to_here, pformat(steps_to_here), cost_to_goal, f_cost, threshold, pformat(self.parent.moves_to_here)))
                self.parent.print_puzzle('PRUNE')
            return False

        if debug:
            log.info("%s: CONT  cost_to_here %d %s, cost_to_goal %d, f_cost %d, threshold %d, moves_to_here %s" %\
                         (self, cost_to_here, pformat(steps_to_here), cost_to_goal, f_cost, threshold, pformat(self.parent.moves_to_here)))
            self.parent.print_puzzle('CONTINUE')

        state = self.state()
        steps = self.steps(state)
        #log.info("FOO: state %s, steps %s" % (pformat(state), pformat(steps)))

        # =================================================
        # If there are steps for a state that means our IDA
        # search is done...woohoo!!
        # =================================================
        if steps:

            # The puzzle is now in a state where it is in the lookup table, we may need
            # to do several lookups to get to our target state though. Use
            # LookupTabele's solve() to take us the rest of the way to the target state.
            LookupTable.solve(self)

            log.info("%s: IDA found match %d steps in, %s, f_cost %d (cost_to_here %d, cost_to_goal %d)" %
                     (self, len(steps_to_here), ' '.join(steps_to_here), f_cost, cost_to_here, cost_to_goal))
            return True

        # ==============
        # Keep Searching
        # ==============
        if f_cost > threshold:
            return False

        # If we have already explored the exact same scenario down another branch
        # then we can stop looking down this branch
        if (cost_to_here, state) in self.explored:
            return False
        self.explored.add((cost_to_here, state))

        moves = self.parent.moves[self.parent.empty_type]

        for step in moves:
            self.parent.move_xyz(step)
            self.ida_count += 1

            if self.ida_search(steps_to_here + [step,], threshold, step,
                               self.parent.state[:],
                               self.parent.moves_to_here[:],
                               copy(self.parent.empty_index),
                               copy(self.parent.empty_type)):
                return True
            else:
                self.parent.state = prev_state[:]
                self.parent.moves_to_here = prev_moves_to_here[:]
                self.parent.empty_index = copy(prev_empty_index)
                self.parent.empty_type = copy(prev_empty_type)

        return False

    def solve(self, max_ida_threshold=100):
        """
        The goal is to find a sequence of moves that will put the puzzle in a state that is
        in our lookup table self.filename
        """

        # This shouldn't happen since the lookup tables are in the repo
        if not self.filename_exists:
            raise PuzzleError("%s does not exist" % self.filename)

        start_time0 = dt.datetime.now()

        state = self.state()
        log.info("%s: ida_stage() state %s vs state_target %s" % (self, state, self.state_target))

        # The puzzle is already in the desired state, nothing to do
        if state == self.state_target:
            log.info("%s: IDA, puzzle is already at the target state" % self)
            return

        # The puzzle is already in a state that is in our lookup table, nothing for IDA to do
        steps = self.steps()

        if steps:
            log.info("%s: IDA, puzzle is already in a state that is in our lookup table" % self)

            # The puzzle is now in a state where it is in the lookup table, we may need
            # to do several lookups to get to our target state though. Use
            # LookupTabele's solve() to take us the rest of the way to the target state.
            LookupTable.solve(self)
            return

        # If we are here (odds are very high we will be) it means that the current
        # puzzle state was not in the lookup table.  We must now perform an IDA search
        # until we find a sequence of moves that takes us to a state that IS in the
        # lookup table.

        # save puzzle state
        self.original_state = self.parent.state[:]
        self.original_moves_to_here = self.parent.moves_to_here[:]
        self.original_empty_index = copy(self.parent.empty_index)
        self.original_empty_type = copy(self.parent.empty_type)

        min_ida_threshold = self.ida_heuristic()
        log.info("%s: IDA threshold range %d->%d" % (self, min_ida_threshold, max_ida_threshold+1))

        for threshold in range(min_ida_threshold, max_ida_threshold+1):
            steps_to_here = []
            start_time1 = dt.datetime.now()
            self.ida_count = 0
            self.explored = set()

            if self.ida_search(steps_to_here, threshold, None,
                               self.original_state[:],
                               self.original_moves_to_here[:],
                               copy(self.original_empty_index),
                               copy(self.original_empty_type)):
                end_time1 = dt.datetime.now()
                log.info("%s: IDA threshold %d, explored %d branches, took %s (%s total)" %
                    (self, threshold, self.ida_count,
                     pretty_time(end_time1 - start_time1),
                     pretty_time(end_time1 - start_time0)))
                return
            else:
                end_time1 = dt.datetime.now()
                log.info("%s: IDA threshold %d, explored %d branches, took %s" %
                    (self, threshold, self.ida_count, pretty_time(end_time1 - start_time1)))

                self.parent.state = self.original_state[:]
                self.parent.moves_to_here = self.original_moves_to_here[:]
                self.parent.empty_index = copy(self.original_empty_index)
                self.parent.empty_type = copy(self.original_empty_type)

        # The only time we will get here is when max_ida_threshold is a low number.  It will be up to the caller to:
        # - 'solve' one of their prune tables to put the puzzle in a state that we can find a moves_to_here for a little more easily
        # - call ida_solve() again but with a near infinite max_ida_threshold...99 is close enough to infinity for IDA purposes
        log.warning("%s: could not find a moves_to_here via IDA within %d steps...will 'solve' a prune table and try again" % (self, max_ida_threshold))

        self.parent.state = self.original_state[:]
        self.parent.moves_to_here = self.original_moves_to_here[:]
        self.parent.empty_index = copy(self.original_empty_index)
        self.parent.empty_type = copy(self.original_empty_type)

        raise NoIDASolution("%s FAILED for state %s" % (self, self.state()))


class PruneTableManhattan(object):

    def __init__(self, parent):
        self.parent = parent

    def cost(self):
        total = 0
        for x in range(1, self.parent.size+1):
            tile_value = self.parent.state[x]

            if tile_value:
                (home_col, home_row) = self.parent.get_col_row(tile_value)
                (curr_col, curr_row) = self.parent.get_col_row(x)
                total += abs(home_row - curr_row) + abs(home_col - curr_col)

        #self.parent.print_puzzle("Manhattan Distance: %d" % total)
        return total


class SliddingTilePuzzle(object):

    moves = {
        'northwest' : ['r', 'd'],
        'northeast' : ['l', 'd'],
        'southwest' : ['r', 'u'],
        'southeast' : ['l', 'u'],
        'north'     : ['r', 'l', 'd'],
        'south'     : ['r', 'l', 'u'],
        'east'      : ['u', 'd', 'l'],
        'west'      : ['u', 'd', 'r'],
        'middle'    : ['u', 'd', 'l', 'r']
    }

    def __init__(self):
        self.state = []
        self.moves_to_here = []

    def __str__(self):
        return "%dx%d" % (self.width, self.height)

    def get_col_row(self, index):

        if index == self.size:
            return (self.width, self.height)

        row = 1
        tmp_index = self.width

        while index > tmp_index:
            row += 1
            tmp_index += self.width

        col = index - ((row-1) * self.width)

        #log.info("index %d, col %d, row %d" % (index, col, row))
        return (col, row)

    def load_solved_puzzle(self, size):
        assert is_square(size), "puzzle must be square (3x3, 4x4, etc), this puzzle has %d entries" % size
        self.size = size
        self.width = int(math.sqrt(self.size))
        self.height = self.width
        self.state = ['placeholder', ]
        self.filename = 'lookup-table-%d.txt' % self.size
        self.filename_gz = self.filename + '.gz'

        if os.path.exists(self.filename_gz) and not os.path.exists(self.filename):
            log.info('gunzip --keep %s' % self.filename_gz)
            subprocess.call(['gunzip', '--keep', self.filename_gz])

        # Find the state_width for the entries in our .txt file
        if os.path.exists(self.filename):
            self.fh_txt = open(self.filename, 'r')

            with open(self.filename, 'r') as fh:
                first_line = next(fh)
                self.line_width = len(first_line)
                (state, steps) = first_line.split(':')
                self.state_width = len(state)
        else:
            self.fh_txt = None
            self.line_width = None
            self.state_width = None

        for x in range(1, self.size):
            self.state.append(x)
        self.state.append(0) # the empty spot
        self.empty_index = self.size
        self.empty_type = 'southeast'

        # Build tuples of which indexes are north, south, east or west edges
        self.north_indexes = []
        for index in range(1, self.width+1):
            self.north_indexes.append(index)
        self.north_indexes = set(self.north_indexes)

        self.south_indexes = []
        for index in range(self.size - self.width + 1, self.size+1):
            self.south_indexes.append(index)
        self.south_indexes = set(self.south_indexes)

        self.west_indexes = []
        for index in range(1, self.size, self.width):
            self.west_indexes.append(index)
        self.west_indexes = set(self.west_indexes)

        self.east_indexes = []
        for index in range(self.width, self.size+1, self.width):
            self.east_indexes.append(index)
        self.east_indexes = set(self.east_indexes)

        #log.info("north indexes: %s" % pformat(self.north_indexes))
        #log.info("south indexes: %s" % pformat(self.south_indexes))
        #log.info("east indexes: %s" % pformat(self.east_indexes))
        #log.info("west indexes: %s" % pformat(self.west_indexes))

    def empty_tile_is_north_edge(self):
        return self.empty_index in self.north_indexes

    def empty_tile_is_south_edge(self):
        return self.empty_index in self.south_indexes

    def empty_tile_is_east_edge(self):
        return self.empty_index in self.east_indexes

    def empty_tile_is_west_edge(self):
        return self.empty_index in self.west_indexes

    def empty_tile_is_northwest_corner(self):
        return self.empty_tile_is_north_edge() and self.empty_tile_is_west_edge()

    def empty_tile_is_northeast_corner(self):
        return self.empty_tile_is_north_edge() and self.empty_tile_is_east_edge()

    def empty_tile_is_southwest_corner(self):
        return self.empty_tile_is_south_edge() and self.empty_tile_is_west_edge()

    def empty_tile_is_southeast_corner(self):
        return self.empty_tile_is_south_edge() and self.empty_tile_is_east_edge()

    def empty_tile_is_middle(self):
        if (self.empty_index not in self.north_indexes and
            self.empty_index not in self.south_indexes and
            self.empty_index not in self.east_indexes and
            self.empty_index not in self.west_indexes):
            return True
        return False

    def set_empty_type(self):

        if self.empty_tile_is_middle():
            self.empty_type = 'middle'

        elif self.empty_tile_is_north_edge():

            if self.empty_tile_is_northwest_corner():
                self.empty_type = 'northwest'

            elif self.empty_tile_is_northeast_corner():
                self.empty_type = 'northeast'

            else:
                self.empty_type = 'north'

        elif self.empty_tile_is_south_edge():

            if self.empty_tile_is_southwest_corner():
                self.empty_type = 'southwest'

            elif self.empty_tile_is_southeast_corner():
                self.empty_type = 'southeast'

            else:
                self.empty_type = 'south'

        elif self.empty_tile_is_east_edge():
            self.empty_type = 'east'

        elif self.empty_tile_is_west_edge():
            self.empty_type = 'west'

        else:
            raise Exception("We should not be here")

    def move_up(self):
        log.debug("move_up")

        if self.empty_tile_is_north_edge():
            raise IllegalMove("move_up requested on %s(%s) which is on north edge" % (self.empty_index, self.empty_type))

        #self.print_puzzle("pre  move_up")
        self.state[self.empty_index] = self.state[self.empty_index - self.width]
        self.state[self.empty_index - self.width] = 0
        self.empty_index -= self.width
        self.set_empty_type()
        self.moves_to_here.append('u')
        #self.print_puzzle("post move_up")

        assert self.state[0] == 'placeholder', "post move_up state %s is invalid" % pformat(self.state)
        #assert len(self.state[1:]) == len(set(self.state[1:])), "Invalid state %s" % pformat(self.state)
        assert self.state[self.empty_index] == 0, "self.state[%d] is %d, it should be 0" % (self.empty_index, self.state[self.empty_index])

    def move_down(self):
        log.debug("move_down")
        #self.print_puzzle("pre  move_down")

        if self.empty_tile_is_south_edge():
            raise IllegalMove("move_down requested on %s(%s) which is on south edge" % (self.empty_index, self.empty_type))

        self.state[self.empty_index] = self.state[self.empty_index + self.width]
        self.state[self.empty_index + self.width] = 0
        self.empty_index += self.width
        self.set_empty_type()
        self.moves_to_here.append('d')
        #self.print_puzzle("post move_down")

        assert self.state[0] == 'placeholder', "post move_down state %s is invalid" % pformat(self.state)
        #assert len(self.state[1:]) == len(set(self.state[1:])), "Invalid state %s" % pformat(self.state)
        assert self.state[self.empty_index] == 0, "self.state[%d] is %d, it should be 0" % (self.empty_index, self.state[self.empty_index])

    def move_left(self):
        log.debug("move_left")
        #self.print_puzzle("pre  move_left")

        if self.empty_tile_is_west_edge():
            raise IllegalMove("move_left requested on %s(%s) which is on west edge" % (self.empty_index, self.empty_type))

        self.state[self.empty_index] = self.state[self.empty_index - 1]
        self.state[self.empty_index - 1] = 0
        self.empty_index -= 1
        self.set_empty_type()
        self.moves_to_here.append('l')
        #self.print_puzzle("post move_left")

        assert self.state[0] == 'placeholder', "post move_up state %s is invalid" % pformat(self.state)
        #assert len(self.state[1:]) == len(set(self.state[1:])), "Invalid state %s" % pformat(self.state)
        assert self.state[self.empty_index] == 0, "self.state[%d] is %d, it should be 0" % (self.empty_index, self.state[self.empty_index])

    def move_right(self):
        log.debug("move_right")
        #self.print_puzzle("pre  move_right")

        if self.empty_tile_is_east_edge():
            raise IllegalMove("move_right requested on %s(%s) which is on east edge" % (self.empty_index, self.empty_type))

        self.state[self.empty_index] = self.state[self.empty_index + 1]
        self.state[self.empty_index + 1] = 0
        self.empty_index += 1
        self.set_empty_type()
        self.moves_to_here.append('r')
        #self.print_puzzle("post move_right")

        assert self.state[0] == 'placeholder', "post move_right state %s is invalid" % pformat(self.state)
        #assert len(self.state[1:]) == len(set(self.state[1:])), "Invalid state %s" % pformat(self.state)
        assert self.state[self.empty_index] == 0, "self.state[%d] is %d, it should be 0" % (self.empty_index, self.state[self.empty_index])

    def move_xyz(self, move_to_make):
        if move_to_make == 'u':
            self.move_up()

        elif move_to_make == 'd':
            self.move_down()

        elif move_to_make == 'l':
            self.move_left()

        elif move_to_make == 'r':
            self.move_right()

        else:
            raise Exception("We should not be here, move '%s'" % move_to_make)

    def randomize(self, size):

        # Start with a solve puzzle
        self.load_solved_puzzle(size)

        # Now perform a bunch of random moves
        for x in range(0, self.size * 10):
            moves = self.moves[self.empty_type]
            moves_len = len(moves)
            move_index = random.randint(0, moves_len-1)
            move_to_make = moves[move_index]
            log.info("%d: empty tile %d (%s) move %s" % (x, self.empty_index, self.empty_type, move_to_make))
            self.move_xyz(move_to_make)

    def print_moves(self):
        print("moves-to-here: %s" % ' '.join(self.moves_to_here))

    def print_puzzle(self, title):
        """
        Print the puzzle as a grid
        """
        print("\n%s\n%s" % (title, len(title) * '='))

        for (index, tile) in enumerate(self.state):

            if index == 0:
                continue

            if index == self.empty_index:
                sys.stdout.write("  ")
            else:
                sys.stdout.write("%2d" % tile)

            if index % self.width == 0:
                sys.stdout.write('\n')
            else:
                sys.stdout.write(' ')

        print('')

    def print_state(self):
        """
        Print the puzzle as a simple string
        """
        print(','.join(map(str, self.state[1:])))

    def load_state(self, state):
        numbers = state.strip().replace(' ', ',').split(',')
        numbers_len = len(numbers)
        assert numbers_len == len(set(numbers)), "puzzle contains a duplicate tile"

        self.load_solved_puzzle(numbers_len)
        self.state = ['placeholder', ]

        for (index, number) in enumerate(numbers):
            assert number.isdigit(), "puzzle contains non-digit tile '%s'" % number
            number = int(number)
            assert number <= self.size-1, "puzzle contains tile %d but max tile value is %d" % (number, self.size-1)
            self.state.append(number)

            if number == 0:
                self.empty_index = index + 1

        self.set_empty_type()

    def solve(self, state):

        if ',' not in state:
            state = ','.join(list(state))

        self.load_state(state)
        st.print_puzzle("Initial Puzzle")

        if self.size == 4:
            lt = LookupTable(self,
                             'lookup-table-4.txt',
                             'all',
                             '1230',
                             linecount=12,
                             size=self.size)
            lt.solve()

        elif self.size == 9:
            lt = LookupTable(self,
                             'lookup-table-9.txt',
                             'all',
                             '123456780',
                             linecount=181440,
                             size=self.size)
            lt.solve()

        elif self.size == 16:
            # pt_manhattan = PruneTableManhattan(self)

            lt_1_6 = LookupTableCumulative(self,
                                           'lookup-table-16-x-1-6.txt',
                                           '1-6',
                                           '123456xxxxxxxxx0',
                                           linecount=57657600,
                                           size=self.size)

            lt_7_12 = LookupTableCumulative(self,
                                           'lookup-table-16-x-7-12.txt',
                                           '7-12',
                                           'xxxxxx789abcxxx0',
                                           linecount=57657600,
                                           size=self.size)

            lt_13_15 = LookupTableCumulative(self,
                                           'lookup-table-16-x-13-15.txt',
                                           '13-15',
                                           'xxxxxxxxxxxxdef0',
                                           linecount=43680,
                                           size=self.size)

            lt = LookupTableIDA(self,
                                'lookup-table-16.txt',
                                'all',
                                '123456789abcdef0',
                                (lt_1_6, lt_7_12, lt_13_15),
                                linecount=42928799, # 24-deep
                                size=self.size)
            lt.solve()

        elif self.size == 25:
            lt_1_2_3 = LookupTable(self,
                                  'lookup-table-25-x-1-2-3.txt',
                                  '1-2-3',
                                  '123xxxxxxxxxxxxxxxxxxxxx0',
                                  linecount=303600,
                                  size=self.size)
            lt_1_2_3.solve()

            lt_4_5_6 = LookupTable(self,
                                  'lookup-table-25-x-4-5-6.txt',
                                  '4-5-6',
                                  '123456xxxxxxxxxxxxxxx0xxx',
                                  linecount=175560,
                                  size=self.size)
            lt_4_5_6.solve()

            lt_11_16_21 = LookupTable(self,
                                  'lookup-table-25-x-11-16-21.txt',
                                  '11-16-21',
                                  '123456xxxxbxxxxgxxxxlxxx0',
                                  linecount=93024,
                                  size=self.size)
            lt_11_16_21.solve()

            # Solve the remaining 16 tiles via the 16-tile solver
            fake_16 = SliddingTilePuzzle()
            state_16 = []
            converter = {
                7 : 1,
                8 : 2,
                9 : 3,
                10 : 4,
                12 : 5,
                13 : 6,
                14 : 7,
                15 : 8,
                17 : 9,
                18 : 10,
                19 : 11,
                20 : 12,
                22 : 13,
                23 : 14,
                24 : 15,
                0 : 0
            }

            for number in (7, 8, 9, 10,
                           12, 13, 14, 15,
                           17, 18, 19, 20,
                           22, 23, 24, 25):
                state_16.append(converter[self.state[number]])

            state_16 = ','.join(map(str, state_16))
            #log.info("state_16: %s" % state_16)
            fake_16.solve(state_16)

            for move in fake_16.moves_to_here:
                self.move_xyz(move)

        elif self.size == 36:
            lt_1_2_3 = LookupTable(self,
                                  'lookup-table-36-x-1-2-3.txt',
                                  '1-2-3',
                                  '123xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx0',
                                  linecount=1413720,
                                  size=self.size)
            lt_1_2_3.solve()

            lt_4_5_6 = LookupTable(self,
                                  'lookup-table-36-x-4-5-6.txt',
                                  '4-5-6',
                                  '123456xxxxxxxxxxxxxxxxxxxxxxxxxxxxx0',
                                  linecount=982080,
                                  size=self.size)
            lt_4_5_6.solve()

            lt_7_13_19 = LookupTable(self,
                                  'lookup-table-36-x-7-13-19.txt',
                                  '7-13-19',
                                  '1234567xxxxxdxxxxxjxxxxxxxxxxxxxxxx0',
                                  linecount=657720,
                                  size=self.size)
            lt_7_13_19.solve()

            lt_25_31 = LookupTable(self,
                                  'lookup-table-36-x-25-31.txt',
                                  '25-31',
                                  'TBD',
                                  linecount=17550,
                                  size=self.size)
            lt_25_31.solve()


            # Solve the remaining 25 tiles via the 25-tile solver
            fake_25 = SliddingTilePuzzle()
            state_25 = []

            converter = {
                8 : 1,
                9 : 2,
                10 : 3,
                11 : 4,
                12 : 5,
                14 : 6,
                15 : 7,
                16 : 8,
                17 : 9,
                18 : 10,
                20 : 11,
                21 : 12,
                22 : 13,
                23 : 14,
                24 : 15,
                26 : 16,
                27 : 17,
                28 : 18,
                29 : 19,
                30 : 20,
                32 : 21,
                33 : 22,
                34 : 23,
                35 : 24,
                0  : 0
            }

            for number in (8, 9, 10, 11, 12,
                           14, 15, 16, 17, 18,
                           20, 21, 22, 23, 24,
                           26, 27, 28, 29, 30,
                           32, 33, 34, 35, 36):
                state_25.append(converter[self.state[number]])

            state_25 = ','.join(map(str, state_25))
            log.info("state_25: %s" % state_25)
            fake_25.solve(state_25)

            for move in fake_25.moves_to_here:
                self.move_xyz(move)

        else:
            raise ImplementThis("Need LookupTable for size %d" % self.size)

        move_verbose = {
            'u' : 'up',
            'd' : 'down',
            'l' : 'left',
            'r' : 'right',
        }

        print("\nSolution")
        print("========")
        for (index, move) in enumerate(self.moves_to_here):
            print("%3d : %s" % (index, move_verbose[move]))
        print("")

        st.print_puzzle("Final Puzzle")

    def lookup_table(self, size, tiles_to_keep=set(), tiles_to_not_move=set(), prune_table=False, max_depth=None):

        # Start with a solved puzzle
        self.load_solved_puzzle(size)
        filename = self.filename

        if tiles_to_keep:
            for index in range(1, self.size):
                if index not in tiles_to_keep and index not in tiles_to_not_move:
                    self.state[index] = 'x'
            filename = 'lookup-table-%d-x-%s.txt' % (self.size, '-'.join(map(str, sorted(list(tiles_to_keep)))))

        if os.path.exists(filename):
            log.warning("%s already exists" % filename)
            return

        init_state = self.state[:]
        init_moves_to_here = self.moves_to_here[:]
        init_empty_index = copy(self.empty_index)
        init_empty_type = copy(self.empty_type)

        self.moves_to_here = []
        lookup_table_states = {}
        lookup_table_states[tuple(self.state[1:])] = ''
        count = 1

        moves_to_make_filename = 'moves_to_make.txt'
        moves_to_make = deque([])
        for move in self.moves[self.empty_type]:
            moves_to_make.append(move)
        log.info("count %d, moves_to_make %d" % (count, len(moves_to_make)))

        while moves_to_make:
            self.state = init_state[:]
            self.moves_to_here = init_moves_to_here[:]
            self.empty_index = init_empty_index
            self.empty_type = init_empty_type
            move_seq = moves_to_make.popleft()
            move_cost = 0
            prev_empty_index = self.empty_index
            move_ok = True

            for move in move_seq:
                self.move_xyz(move)

                # This is a prune table
                if (prune_table and
                    self.state[prev_empty_index] != 'x' and
                    self.state[prev_empty_index] in tiles_to_keep):
                    move_cost += 1

                if self.state[prev_empty_index] in tiles_to_not_move:
                    move_ok = False
                    break

                prev_empty_index = self.empty_index

            if move_ok and tuple(self.state[1:]) not in lookup_table_states:

                # This is a prune table
                if prune_table:
                    lookup_table_states[tuple(self.state[1:])] = move_cost

                # This is a normal LookupTable so reverse the steps and store the first one
                # We only need to store the first one....this allows us to save some disk space
                else:
                    lookup_table_states[tuple(self.state[1:])] = reverse_moves(self.moves_to_here[:])[0]

                # Save some memory...write new moves_to_make to a file
                to_add = []
                for move in self.moves[self.empty_type]:
                    new_move_seq = list(move_seq) + list(move)

                    if max_depth is None or len(new_move_seq) <= max_depth:
                        to_add.append(new_move_seq)

                if to_add:
                    with open(moves_to_make_filename, 'a') as fh:
                        for new_move_seq in to_add:
                            fh.write("%s\n" % ''.join(new_move_seq))

                count += 1

                if count % 100000 == 0:
                    log.info("count %d, moves_to_make %d" % (count, len(moves_to_make)))

            if not moves_to_make:
                if os.path.exists(moves_to_make_filename):
                    with open(moves_to_make_filename, 'r') as fh:
                        for line in fh:
                            line = line.strip()
                            moves_to_make.append(list(line))
                    os.unlink(moves_to_make_filename)


        # =======
        # writing
        # =======
        # Write the lookup_table_states content to a file
        log.info("writing %s" % filename)

        with open(filename, 'w') as fh:
            extended_hex = {
                16 : 'g',
                17 : 'h',
                18 : 'i',
                19 : 'j',
                20 : 'k',
                21 : 'l',
                22 : 'm',
                23 : 'n',
                24 : 'o',
                25 : 'p',
                26 : 'q',
                27 : 'r',
                28 : 's',
                29 : 't',
                30 : 'u',
                31 : 'v',
                32 : 'w',
                33 : '-', # we alredy used 'x' for tiles we are not tracking
                34 : 'y',
                35 : 'z',
            }

            for (key, value) in lookup_table_states.items():

                # Write numbers in hex
                foo = []
                for number in key:
                    if isinstance(number, int):
                        if number <= 15:
                            foo.append("%x" % number)
                        else:
                            foo.append(extended_hex[number])
                    else:
                        foo.append(number)
                state = ''.join(foo)
                fh.write("%s:%s\n" % (state, value))


        # =======
        # sorting
        # =======
        # Sort the file so we can binary search the file later
        if not os.path.exists('./tmp/'):
            os.mkdir('./tmp/')

        log.info("sort %s" % filename)
        cores_for_sort = 4
        subprocess.check_output("sort --parallel=%d --temporary-directory=./tmp/ --output=%s %s" %
                                (cores_for_sort, filename, filename),
                                shell=True)

        # =======
        # padding
        # =======
        # Make all lines the same length so we can binary search the file later
        log.info("./utils/pad_lines.py %s" % filename)
        subprocess.check_output(['./utils/pad_lines.py', filename])


if __name__ == '__main__':

    # logging.basicConfig(filename='rubiks-rgb-solver.log',
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)7s: %(message)s')
    log = logging.getLogger(__name__)

    # Color the errors and warnings in red
    logging.addLevelName(logging.ERROR, "\033[91m  %s\033[0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.WARNING, "\033[91m%s\033[0m" % logging.getLevelName(logging.WARNING))

    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--solve', help='Solve puzzle', type=str, default=None)
    action.add_argument('--random', help='Create random puzzle', type=int, default=None)
    action.add_argument('--lookup-table', help='Build lookup-table', type=int, default=None)
    args = parser.parse_args()

    st = SliddingTilePuzzle()

    try:
        if args.solve:
            st.solve(args.solve)

        elif args.random:
            st.randomize(args.random)
            st.print_puzzle()

        elif args.lookup_table:

            if args.lookup_table == 4:
                st.lookup_table(args.lookup_table, prune_table=False)

            elif args.lookup_table == 9:
                st.lookup_table(args.lookup_table, prune_table=False)

            elif args.lookup_table == 16:

                # 20-deep
                # - took about 5 minutes
                # - 3.4 million entries
                #
                # 22-deep
                # - took about 27 minutes
                # - 12.3 million entries
                st.lookup_table(args.lookup_table, prune_table=False, max_depth=24)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((1, 2, 3, 4, 5, 6)), prune_table=True)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((7, 8, 9, 10, 11, 12)), prune_table=True)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((13, 14, 15)), prune_table=True)

            elif args.lookup_table == 25:
                st.lookup_table(args.lookup_table, tiles_to_keep=set((1, 2, 3)), prune_table=False)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((4, 5, 6)), tiles_to_not_move=set((1, 2, 3)), prune_table=False)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((11, 16, 21)), tiles_to_not_move=set((1, 2, 3, 4, 5, 6)), prune_table=False)


            elif args.lookup_table == 36:
                st.lookup_table(args.lookup_table, tiles_to_keep=set((1, 2, 3)), prune_table=False)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((4, 5, 6)), tiles_to_not_move=set((1, 2, 3)), prune_table=False)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((7, 13, 19)), tiles_to_not_move=set((1, 2, 3, 4, 5, 6)), prune_table=False)
                st.lookup_table(args.lookup_table, tiles_to_keep=set((25, 31)), tiles_to_not_move=set((1, 2, 3, 4, 5, 6, 7, 13, 19)), prune_table=False)

            else:
                raise ImplementThis("Add lookup-table support for %d" % args.lookup_table)

        else:
            raise Exception("We should not be here")

    except PuzzleError:
        st.print_moves()
        st.print_puzzle("Error")
        raise
