from __future__ import division

def to_time_str(s):
    """
    Convert the time from second to readable-format string.
    """
    s = int(round(s))
    seconds = s % 60
    minutes = (s // 60) % 60
    hours = ((s // 60) // 60) % 24
    days = ((s // 60) // 60) // 24
    # get the string
    res = ""
    if days > 0:
        res += "%dd " % days
    if hours > 0:
        res += "%02d:" % hours
    res += "%02d:%02d" % (minutes, seconds)
    return res
