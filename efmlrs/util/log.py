"""
Collection of small scripts for creating log file.
"""
import inspect
from datetime import datetime

def log_init(name):
    name += "_compression.log"
    if type(log_init.logfile) == bool:
        log_init.logfile = open(name, "w")


log_init.logfile = False
log_init.search = False


def log():
    return log_init.logfile


def log_time():
    timestamp = datetime.now()
    log().write(str(timestamp) + " ")

def log_module():
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__name__
    log_time()
    log().write("*** " + filename + " ***\n")


def log_close():
    if type(log()) != bool:
        log().close()


def log_add():
    return "   [+] "


def log_del():
    return "   [-] "


def log_keep():
    return "   [>] "


def log_strip(rea):
    return rea.strip('"')


def merge_reaction_names(R1_keep, R2_remove):
    return R1_keep + ":" + R2_remove


def log_merge(remove, keep, keep_factor):
    log_time()
    log().write("merging ...\n")
    log_time()
    log().write(log_del() + log_strip(remove) + " factor: " + str(keep_factor) + " \n")
    log_time()
    log().write(log_add() + log_strip(keep) + " \n")
    if log_init.search is False:
        keep = merge_reaction_names(keep, remove)
        log_time()
        log().write(log_keep() + log_strip(keep) + " \n")
        return True, [keep]
    log_time()
    log().write(log_keep() + log_strip(keep) + " \n")
    return True, []


def log_merge_many(remove, keep, remove_factor, factors, force=False):
    index = 0
    log_time()
    log().write("merging " + str(len(keep)) + " ...\n")
    log_time()
    log().write(log_del() + log_strip(remove) + " factor: " + str(remove_factor) + " \n")
    names = []

    for rea in keep:
        log_time()
        log().write(log_add() + log_strip(rea) + " factor: " + str(factors[index]) + " \n")
        if log_init.search is False:
            rea = merge_reaction_names(rea, remove)
            names.append(rea)
        log_time()
        log().write(log_keep() + log_strip(rea) + "\n")
        index += 1
    return True, names


def log_delete_meta(meta):
    log_time()
    log().write("deleting metabolite " + log_strip(meta) + " \n")


def log_delete_rea(rea):
    log_time()
    log().write("deleting ... \n")
    log_time()
    log().write(log_del() + log_strip(rea) + " \n")


def log_delete(rea1, rea2):
    log_time()
    log().write("deleting ... \n")
    log_time()
    log().write(log_del() + log_strip(rea1) + " \n")
    log_time()
    log().write(log_del() + log_strip(rea2) + " \n")


def log_merge_rea(rea):
    log_time()
    log().write("merging single \n")
    log_time()
    log().write(log_add() + log_strip(rea) + " \n")
    log_time()
    log().write(log_keep() + log_strip(rea) + " \n")
