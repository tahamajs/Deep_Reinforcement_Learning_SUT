# Author: Taha Majlesi - 810101504, University of Tehran
class BaseCritic(object):
    def update(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        raise NotImplementedError
