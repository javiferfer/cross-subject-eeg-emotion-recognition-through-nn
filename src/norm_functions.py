import numpy as np

no_sessions = 3
no_participants = 15
no_channels = 62
no_features = 5


def normalization(data, no_videos):

    bandpower_seed_normalization = []

    # Number of features
    no_features = int(len(data[0])/no_channels)

    for i in range(no_participants):
        for j in range(no_sessions):
            for k in range(no_videos):
                normalized_data = []
                for l in range(no_channels):
                    for m in range(no_features):
                        max_number = np.max(data[:, l*no_features+m])
                        min_number = np.min(data[:, l*no_features+m])
                        feature = data[i*no_sessions*no_videos+j*no_videos + k, l * no_features + m]
                        new_feature = (feature-min_number)/(max_number - min_number)
                        normalized_data.append(new_feature)
                bandpower_seed_normalization.append(normalized_data)

    bandpower_seed_normalization = np.array(bandpower_seed_normalization)
    return bandpower_seed_normalization


def normalization_per_participant_session(data, no_videos):

    bandpower_seed_normalization = []

    # Number of features
    no_features = int(len(data[0])/no_channels)

    for i in range(no_participants):
        for j in range(no_sessions):
            for k in range(no_videos):
                normalized_data = []
                for l in range(no_channels):
                    for m in range(no_features):
                        max_number = np.max(data[i*no_sessions*no_videos + j * no_videos:
                                                 i*no_sessions*no_videos + (j + 1) * no_videos,
                                                 l*no_features+m])
                        min_number = np.min(data[i*no_sessions*no_videos + j * no_videos:
                                                 i*no_sessions*no_videos + (j + 1) * no_videos,
                                                 l*no_features+m])
                        feature = data[i*no_sessions*no_videos+j*no_videos + k, l * no_features+m]
                        new_feature = (feature-min_number)/(max_number - min_number)
                        normalized_data.append(new_feature)
                bandpower_seed_normalization.append(normalized_data)

    bandpower_seed_normalization = np.array(bandpower_seed_normalization)
    return bandpower_seed_normalization
