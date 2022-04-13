from typing import Sized


class Buffer(Sized):
    def __init__(self):
        pass

    def sample_buffer(self, batch_size: int):
        pass

    def store_transition(self, *transition):
        pass

    def __len__(self):
        pass
