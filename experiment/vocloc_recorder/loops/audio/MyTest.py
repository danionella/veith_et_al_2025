from queue import Empty


class MyTest:
    """
    """

    def __init__(self, conf):
        print("My Test init")

    def loop(self, q, qTrig, q_audio2triggered):
        try:
            # get data
            chunk = q.get(timeout=0.5)  # for now without timestamps
            data = chunk[:-1, :]  # remove trigger channel
            print("Got chunk of shape ", chunk.shape)

            # Can trigger the upon_trigger function in the triggered folder here:
            #   qTrig.set()

        except Empty:
            pass

    def close(self):
        print(f"My Test done")
