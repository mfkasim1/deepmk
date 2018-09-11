import time
from deepmk.utils.mktime import to_time_str

class ProgressDisplay:
    def __init__(self, len_progress_bar=20):
        self.len_progress_bar = 20
        self.reset()

    def reset(self):
        self.progress_printed = False
        self.start_time = time.time()

    def show(self, completed, total):
        progress = 1.0 * completed / total

        # delete the row
        if self.progress_printed:
            print("\033[F" + (" "*1) + "\033[F")

        progress_str = "=" * (int(progress * self.len_progress_bar))
        progress_str += " " * (self.len_progress_bar - len(progress_str))

        # estimate the time
        elapsed_time = time.time() - self.start_time
        remaining_time = elapsed_time / progress * (1.0 - progress)

        # print the progress
        print("Progress: [%s] %8d/%8d. " \
              "Elapsed: %s. "\
              "ETA: %s" % \
               (progress_str, completed, total,
               to_time_str(elapsed_time),
               to_time_str(remaining_time)))
        self.progress_printed = True
