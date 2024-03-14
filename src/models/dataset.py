class CUFEDImportanceDataset(data.Dataset):
    def __init__(self, data_path, album_list, transforms, args=None) -> None:
        super(CUFEDImportanceDataset, self).__init__()
        self.args = args
        self.data_path = data_path
        self.albums = np.loadtxt(album_list, dtype='str', delimiter='\n')
        self.transforms = transforms
        self.cls_dict = json.load(open(args.event_type_pth))

        self.labels_str = ['Architecture', 'BeachTrip', 'Birthday', 'BusinessActivity',
                           'CasualFamilyGather', 'Christmas', 'Cruise', 'Graduation', 'GroupActivity',
                           'Halloween', 'Museum', 'NatureTrip', 'PersonalArtActivity',
                           'PersonalMusicActivity', 'PersonalSports', 'Protest', 'ReligiousActivity',
                           'Show', 'Sports', 'ThemePark', 'UrbanTrip', 'Wedding', 'Zoo']

        self.num_cls = len(self.labels_str)
        self.classes_to_idx = {}
        for i, cls in enumerate(self.labels_str):
            self.classes_to_idx[cls] = i

        self.data = ImageFolder(data_path, transforms)

        self.scores_dict = json.load(open(args.image_importance_pth))

        # print(self.scores_dict.values())
        self.scores = {img[1]: img[2]
                       for imgs in self.scores_dict.values() for img in imgs}

        self.index_to_classes = {v: k for k,
                                 v in self.data.class_to_idx.items()}

        self.max_score = 2
        self.min_score = -2