trainselectlocationlist = [  # define which location to use
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]

testselectlocationlist = [
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]


def get_split_locations(split):
    if split == "train":
        return trainselectlocationlist
    if split == "test":
        return testselectlocationlist
    raise NotImplementedError
