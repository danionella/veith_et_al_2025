import time


class MyTest:
    def __init__(self, conf):
        print("Class is initialized.")

    def loop(self):
        """
        This function is executed continuously
        """
        time.sleep(3)
        print("Function is executed.")
        return

    def close(self):
        pass
