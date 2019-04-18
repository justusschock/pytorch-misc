import torch


def euclidian_distance(x1, x2):
    # add small eps to avoid sqrt(0)
    return torch.sqrt((x1 - x2).sum(-1) **2 + 1e-6)


def make_onehot(labels, n_classes):
    labels = labels.view(-1, 1).long()

    labels_onehot = torch.zeros(labels.size(0), n_classes,
                                device=labels.device, dtype=torch.float)
    labels_onehot.scatter_(1, labels, 1)
    return labels_onehot


class KNN(torch.nn.Module):
    """
    Class implementing a basic KNN with c
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor, k,
                 distance_fn=euclidian_distance, labels_onehot=False):
        """

        Parameters
        ----------
        features : torch.Tensor
            the features of the trainset, must be of shape (1, N, F) or
            (N, F), where N is the number of samples and
            F is the number of features
        labels : torch.Tensor
            the labels corresponding to the given ``features``.
            Can be of shape Nx1 or N if they aren't one-hot encoded
            or of shape NxC if they are with (C being the number of classes)
        k : int
            number specifying how many of the nearest neighbors to consider
            during voting
        distance_fn : function
            function to calculate the distance; output shape must match the
            input shape with a squeezed last dimension
            default: euclidian_disctances
        labels_onehot : bool
            whether labels are given one-hot encoded or not; default: False
        """
        super().__init__()

        # add batch dimension
        if features.ndimension() == 2:
            features = features.unsqueeze(0)
        elif features.ndimension() > 3:
            raise ValueError("features must have a dimensionality of either "
                             "2 or 3, but got %d" % features.ndimension())

        self.register_buffer("features", features)

        # convert labels to onehot if necessary

        if not labels_onehot:
            labels = make_onehot(labels, labels.max().item() + 1)

        # add additional dimension for indexing
        labels = labels.unsqueeze(0)
        self.register_buffer("labels", labels)
        self.k = k
        self.distance_fn = distance_fn
        self.labels_onehot = labels_onehot

    def forward(self, x: torch.Tensor):
        """
        Predicts a batch of samples.

        Parameters
        ----------
        x : torch.Tensor
            can be a single sample of shape (F) or (1, F) or a
            batch of samples shaped (NxF)

        Returns
        -------
        torch.Tensor
            class indices: will be onehot encoded if ``labels_onehot``
            was set during class initialization
            for a single sample, an additional batch dimension of
            1 will be added during inference and squeezed afterwards
        """

        squeeze_first = True
        if x.ndimension() == 1:
            # add batch and feature dimension
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze_first = True
        elif x.ndimension() == 2:
            # add feature dimension
            x = x.unsqueeze(1)

        # calculate distances to each data sample
        distances = self.distance_fn(x, self.features)

        # get indices for first k samples
        indices = torch.argsort(distances, -1)[..., : self.k]

        # extract class indices for these indices
        labels = self.labels
        labels = labels.expand(x.size(0), *labels.shape[1:])

        class_votings = labels[:, indices]

        class_idxs = torch.argmax(class_votings.sum(-2), dim=-1)

        if self.labels_onehot:
            class_idxs = make_onehot(class_idxs, labels.size(-1))

        # squeeze first batch dimension for single sample input
        # to ensure consistent shape w.r.t. input
        if squeeze_first:
            class_idxs = class_idxs.squeeze(0)

        return class_idxs


if __name__ == '__main__':
    features_2d = torch.rand(100, 10)
    features_3d = torch.rand(1, 100, 10)

    labels = torch.randint(0, 10, size=(100,))
    labels_onehot = make_onehot(labels, 10)
    input_single = torch.rand(10)
    input_batch = torch.rand(5, 10)

    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    k = 3

    for device in devices:
        knn = KNN(features_2d, labels, k, labels_onehot=False).to(device)

        pred_single = knn(input_single.to(device))
        pred_batch = knn(input_batch.to(device))

        knn = KNN(features_2d, labels_onehot, k,
                  labels_onehot=True).to(device)

        pred_single_onehot = knn(input_single.to(device))
        pred_batch_onehot = knn(input_batch.to(device))

        if (make_onehot(pred_batch, 10) == pred_batch_onehot).all():
            print("Works for 2d batched on %s" % str(device))

        if (make_onehot(pred_single, 10) == pred_single_onehot).all():
            print("works for 2d single on %s" % str(device))

        knn = KNN(features_3d, labels, k, labels_onehot=False).to(device)

        pred_single = knn(input_single.to(device))
        pred_batch = knn(input_batch.to(device))

        knn = KNN(features_3d, labels_onehot, k,
                  labels_onehot=True).to(device)

        pred_single_onehot = knn(input_single.to(device))
        pred_batch_onehot = knn(input_batch.to(device))

        if (make_onehot(pred_batch, 10) == pred_batch_onehot).all():
            print("Works for 3d batched on %s" % str(device))

        if (make_onehot(pred_single, 10) == pred_single_onehot).all():
            print("works for 3d single on %s" % str(device))
