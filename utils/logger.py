import os

import numpy as np
import pickle
from typing import List

import matplotlib.pyplot as plt
import curses
import sys


class LoggerGroup:
    def __init__(self, title="LoggerGroup"):
        """
        This class will create a dictionary for each variable which contains 3 keys
        - 'f_hist' : A list which store all value steps.
        - 'epchs' : A list which store an average of all steps in each epochs
        - 'each_epch' : Store all step in each epochs (This list will be clear
                        when the flush_epoch_all() has been called)
        """
        self.__dict = {}
        self.title = title

    def add_var(self, *keys):
        for e_k in keys:
            self.__dict[e_k] = {
                'f_hist': [],
                'epchs': [],
                'each_epch': [],
                'sub_each_epch': []
            }

    def get_latest_step(self):
        """
        :return: Return the last variable step of all available variable in this group
        """
        r_obj = {}
        for e_k in self.__dict.keys():
            if len(self.__dict[e_k]['each_epch']) > 0:  # if 'each_epch' steps is not empty return the latest step
                r_obj[e_k] = self.__dict[e_k]['each_epch'][-1]
            elif len(self.__dict[e_k]['epchs']) > 0:  # but if it's no step available, return 'epchs' instead
                r_obj[e_k] = self.__dict[e_k]['epchs'][-1]
            else:  # but if there is no step at all, then return none... let the reporter do the rest
                r_obj[e_k] = None
        return r_obj

    def collect_sub_step(self, key: str, value: float):
        if key not in self.__dict.keys():
            self.add_var(key)
        self.__dict[key]['sub_each_epch'].append(value)

    def flush_sub_step_all(self):
        for e_k in self.__dict.keys():
            sub_each_epch = self.__dict[e_k]['sub_each_epch']
            if len(sub_each_epch) > 0:
                # self.__dict[e_k]['f_hist'] += sub_each_epch
                self.__dict[e_k]['each_epch'].append(sum(sub_each_epch) / len(sub_each_epch))
                self.__dict[e_k]['sub_each_epch'] = []

    def collect_step(self, key: str, value: float):
        if key not in self.__dict.keys():
            self.add_var(key)
        self.__dict[key]['each_epch'].append(value)

    def flush_step_all(self):
        for e_k in self.__dict.keys():
            each_epch = self.__dict[e_k]['each_epch']
            if len(each_epch) == 0:  # Skip the key if there is nothing to flush
                continue
            else:
                self.__dict[e_k]['f_hist'] += each_epch
                self.__dict[e_k]['epchs'].append(sum(each_epch) / len(each_epch))
                self.__dict[e_k]['each_epch'] = []

    def collect_epch(self, key: str, value: float):
        if key not in self.__dict.keys():
            self.add_var(key)
        self.__dict[key]['epchs'].append(value)

    def report(
            self,
            show_fig=True,
            save_fig=False,
            save_details_txt=False,
            log_scale=False
    ):
        """
        This method will report all the available log variable
        :param show_fig: set to show figure
        :param save_fig: set to save all figure
        :param save_details_txt: set to save additional details in text file
        :param log_scale: set to save a plot figure in a log scale
        """
        key_list = list(self.__dict.keys())
        self.plot(key_list, log_scale, save_fig, show_fig, full_hist=False, show_legend=True)
        self.plot(key_list, log_scale, save_fig, show_fig, full_hist=True, show_legend=True)
        if save_details_txt:
            f = open(self.title + "_details.txt", "w")
            f.write(self.summary())
            f.close()

    def plot(self,
             key_list: list,
             log_scale: bool = False,
             save_fig: bool = False,
             show_fig: bool = False,
             full_hist: bool = False,
             show_legend: bool = False
             ):
        """
        This function will plot the history of log on the given keys in a single figure.
        :param key_list The given keys that will use to plot all of those value on a figure
        :param log_scale Specify to plot in log scale or not
        :param save_fig Save the figure in to a file
        :param show_fig Show plotted figure
        :param full_hist Plot the full history (True) or plot the epochs history (False)
        :param show_legend Specify to show plot legend or not
        """
        hist_mode = 'f_hist' if full_hist else 'epchs'
        fig_title = self.title + "_" + hist_mode
        plt.title(fig_title)
        for e_k in key_list:
            arr = self.__dict[e_k][hist_mode]
            if len(arr) > 0:
                if log_scale:
                    arr = np.log(np.add(arr, 1))
                plt.plot(arr, label=e_k)
        if show_legend:
            plt.legend()
        if save_fig:
            f_name = self.title + "-"
            f_name += 'f_hist' if full_hist else 'epchs'
            f_name += '_log_scaled' if log_scale else ''
            plt.savefig("./%s" % f_name)
        if show_fig:
            plt.show()
        plt.clf()

    def summary(self, key_list: list = None, log_scale: bool = False, full_hist: bool = False):
        """
        This method will return summary string
        """
        head = "<===[[ " + self.title + " ]]===> {"
        dtx = head
        key_list = list(self.__dict.keys()) if key_list is None else key_list
        for e_k in key_list:
            dtx += self.__gen_summary(key=e_k, indent=0, log_scale=log_scale, full_hist=full_hist)
        return dtx + "\n}"

    def __gen_summary(self, key, indent=0, log_scale: bool = False, full_hist: bool = False):
        arr = self.__dict[key]['f_hist'] if full_hist else self.__dict[key]['epchs']
        ind = '\t' * indent
        if log_scale:
            arr = np.log(np.add(arr, 1))
        if len(arr) > 0:
            mn_v = min(arr)
            mx_v = max(arr)
            lst_v = arr[-1]
            txt = "\n" + ind + "{}\n" + ind + "\t> min:{:.2f}\n" + ind + "\t> max:{:.2f}\n" + ind + "\t> last:{:.2f}"
            return txt.format(key, mn_v, mx_v, lst_v)
        else:
            return ""

    def export_file(self, filename: str):
        if ".lggr" not in filename:
            filename = filename + ".lggr"
        f = open(filename, 'wb')
        obj = {
            'title': self.title,
            'dict': self.__dict
        }
        pickle.dump(obj, f)

    def load_file(self, filename: str):
        f = open(filename, 'rb')
        obj = pickle.load(f)
        self.title = obj['title']
        self.__dict = obj['dict']


class Reporter:
    def __init__(self, *logger_groups):
        self.l_g: List = list(logger_groups)
        self.line_count = 0  # This variable only use in classic reporter
        self.exp_name = "**An Experiment**"
        self.sum_desc = ""

    def __init_stdscr(self):
        try:
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.use_classic_report = False
        except curses.error:
            print("<I> Cannot identify terminal. Using classic reporter")
            self.use_classic_report = True

    def classic_reporter(self, classic_mode: bool):
        if classic_mode:
            self.use_classic_report = True
            self.stop()
        else:
            self.__init_stdscr()
        return self

    def review(self):
        """
        Review experiment details
        """
        try:
            r_head = "Experiment name: " + self.exp_name
            r_dash = "-" * (len(r_head) + 4)
            r_desc = self.sum_desc
            print(r_head + "\n" + r_dash)
            print(r_desc + "\n")
            print("\tPress Enter to start training")
            input("\tor Ctrl-C to abort...")
            self.__init_stdscr()
            return True
        except KeyboardInterrupt:
            print("\n<I> : User rejected the experiment.")
            return False

    def append_logger_list(self, l_g_list: List):
        self.l_g += l_g_list

    def append_logger_dict(self, l_g_dict: dict):
        for ek in l_g_dict.keys():
            self.l_g += [l_g_dict[ek]]

    def report(self, epch=None, b_i=None, b_all=None):
        """

        :param epch: number of epoch that training
        :param b_i: number of batch that in training
        :param b_all: number of all batch in the data loader
        :return:
        """

        # Build epoch tracker string
        epch_str = "??" if epch is None else "{:4}".format(epch)
        b_i_str = "??" if b_i is None else "{:3}".format(b_i)
        b_all_str = "??" if b_all is None else "{:3}".format(b_all)
        report_str = 'Epoch [%s|%s/%s] >> ' % (epch_str, b_i_str, b_all_str)

        if self.use_classic_report:
            for i, e_lg in enumerate(self.l_g):
                report_str += e_lg.title + ' > '
                vars_step = e_lg.get_latest_step()
                for e_v in vars_step.keys():
                    if vars_step[e_v] is not None:
                        report_str += "{}[{:8.4f}],".format(e_v, vars_step[e_v])
                    else:
                        report_str += "{}[N/A],".format(e_v)
                report_str = report_str[0:-1] + ' | '
            sys.stdout.write('\r' + report_str[0:-3])

        else:
            line_idx = 0
            if epch is not None and b_i is not None and b_all is not None:
                self.stdscr.addstr(line_idx, 0, report_str)
                line_idx += 1

            # Iter on each group
            for i, e_lg in enumerate(self.l_g):
                title = e_lg.title
                self.stdscr.addstr(line_idx, 0, "{}".format(title))
                line_idx += 1
                vars_step = e_lg.get_latest_step()
                var_report = " > "
                for e_v in vars_step.keys():
                    if vars_step[e_v] is not None:
                        var_report += "{}[{:10.4f}], ".format(e_v, vars_step[e_v])
                    else:
                        var_report += "{}[N/A], ".format(e_v)
                self.stdscr.addstr(line_idx, 0, var_report[0:-2])
                line_idx += 1
            self.stdscr.refresh()

    def set_experiment_name(self, exp_name):
        self.exp_name = exp_name

    def set_summary_description(self, txt):
        self.sum_desc = txt

    def append_summary_description(self, txt):
        self.sum_desc += txt + " "

    def write_summary(self, path, f_name: str = "summary.txt", log_scale: bool = False, full_hist: bool = False):
        txt = self.exp_name + "\n" + ("-" * (len(self.exp_name) + 5)) + "\n"
        txt += self.sum_desc + "\n\nHere are experiment results.\n"
        for e_lg in self.l_g:
            txt += e_lg.summary(log_scale=log_scale, full_hist=full_hist) + "\n"
        f = open(os.path.join(path, f_name), "w")
        f.write(txt)
        f.close()

    def stop(self):
        if not self.use_classic_report:
            try:
                del self.stdscr
                curses.echo()
                curses.nocbreak()
                curses.endwin()
            except AttributeError:
                print("<I> : Training script was done while screen not active.")


# if __name__ == '__main__':
#     a = LoggerGroup("Untitled")
#     b = LoggerGroup("Untitled2")
#     r = Reporter(a, b)
#     for i in range(16):
#         a.collect_epch('D_slope', i * 0.57)
#         a.collect_epch('Q_slope', i)
#         b.collect_epch('Anti_Q', (i ** 1.4) * (-0.86))
#         b.collect_epch('Anti_M', (i ** 1.4) - i * (-0.86))
#     r.write_summary('./', False, False)
#     a.report(show_fig=True)
#     b.plot(key_list=['Anti_Q'], show_fig=True, show_legend=True)
