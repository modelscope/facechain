import torch

class Local_Face_Rep_Input:
    def __init__(self):
        self.set = False

    def prepare(self, local_fac_rep):

        self.set = True
        face = local_fac_rep

        self.batch, self.token_num, self.dim = face.shape

        self.device = face.device
        self.dtype = face.dtype

        return face
        
    def get_null_input(self, batch=None, device=None, dtype=None):

        """
        Guidance for training (drop) or inference,
        please define the null input for the grounding tokenizer
        """

        assert self.set, "not set yet, cannot call this function"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        face = torch.zeros(self.batch, self.token_num, self.dim).type(dtype).to(device)

        return face