import pandas as pd
from torch.utils.data import Dataset

from templates import template_1, template_2, template_3, template_4


class SymileDataset(Dataset):
    """
    TODO: add comments
    """
    def __init__(self, args):
        # TODO: you may not even need the template information after all...
        # but keep it in for now in case you do. if you don't need, then
        # take out of templates.py file, too.
        t1 = template_1(args)[["text", "audio_path", "image_path", "template"]]
        t2 = template_2(args)[["text", "audio_path", "image_path", "template"]]
        t3 = template_3(args)[["text", "audio_path", "image_path", "template"]]
        t4 = template_4(args)[["text", "audio_path", "image_path", "template"]]
        self.df = pd.concat([t1, t2, t3, t4], ignore_index=True)

    def __len__(self):
        """
        Compute length of the dataset.

        Returns:
            (int): dataset size.
        """
        return len(self.df)

    def get_audio(self, path):
        pass

    def get_image(self, path):
        pass

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns: TODO
        """
        template = self.df.iloc[idx].template
        text = self.df.iloc[idx].text
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        image = self.get_image(self.df.iloc[idx].image_path)

        # TODO: somehow put these in the forms that the encoders expect

        return {"text": text, "audio": audio, "image": image, "template": template}

