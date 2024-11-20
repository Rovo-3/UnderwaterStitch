import cv2
import multiprocessing
import time

from DetectMatchConfidence import DetectMatchConfidence


class ProcessPairingImage:
    def __init__(self, method, dmc):
        self.dmc = dmc
        self.method = method

    def keypoints_to_tuple(self, keypoints):
        return [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in keypoints]

    def tuple_to_keypoints(self, keypoint_tuples):
        return [
            cv2.KeyPoint(x, y, size, angle) for (x, y, size, angle) in keypoint_tuples
        ]

    def calculate_ranges(self, totalimage, num_processes):
        # Calculate approximate chunk size for each process
        chunk_size = totalimage // num_processes
        remainder = totalimage % num_processes

        ranges = []
        start = 0

        for i in range(num_processes):
            # Add 1 to chunk size if there's a remainder to distribute evenly
            end = start + chunk_size + (1 if i < remainder else 0) - 1
            ranges.append((start, end))
            start = end + 1

        return ranges

    def process_image_pair(
        self, startI, endI, totalimage, arrimgname, arrdescriptors, arrkeypoints
    ):
        arrconfidence = []
        matxConf = []
        arrkeypoints = [self.tuple_to_keypoints(kp) for kp in arrkeypoints]

        for i in range(startI, endI + 1):
            arrconfidence = []
            for j in range(totalimage):
                if i >= len(arrimgname) or j >= len(arrimgname):
                    raise IndexError(f"Index out of range: i={i}, j={j}")

                if arrimgname[i] == arrimgname[j]:
                    confidence = 0
                    arrconfidence.append(int(confidence))
                    continue
                    # return arrconfidence

                # KNN
                if self.method == "knn":
                    try:
                        matches = self.dmc.BFMatchKNN(
                            arrdescriptors[i], j, arrdescriptors
                        )
                        confidence = self.dmc.knnConfidenceMatch(
                            matches, arrkeypoints[i], j, arrkeypoints
                        )
                    except:
                        confidence = 0

                # BF
                elif self.method == "bf":
                    matches = self.dmc.BFMatch(
                        arrdescriptors[i], j, arrdescriptors, nmatches=2000
                    )
                    confidence = self.dmc.findConfidenceMatch(
                        matches, arrkeypoints[i], j, arrkeypoints
                    )

                arrconfidence.append(int(confidence * 1000))
            matxConf.append(arrconfidence)

        return matxConf

    def imageOrderByConf_Multi(
        self, totalimage, arrimgname, arrdescriptors, arrkeypoints
    ):

        arrimgname = tuple(arrimgname)
        arrdescriptors = tuple(arrdescriptors)
        arrkeypoints = [self.keypoints_to_tuple(kp) for kp in arrkeypoints]

        numprocesses = multiprocessing.cpu_count()
        # numprocesses = 4
        print(f"Utilizing {numprocesses} processes.")

        ranges = self.calculate_ranges(totalimage, numprocesses)
        print(ranges)

        with multiprocessing.Pool(processes=numprocesses) as pool:
            # Map each range to a separate process
            results = pool.starmap(
                self.process_image_pair,
                [
                    (start, end, totalimage, arrimgname, arrdescriptors, arrkeypoints)
                    for start, end in ranges
                ],
            )
            # results = pool.starmap(process_image_pair, ranges)

        matxConf = []
        for res in results:
            matxConf.append(res)

        flat_array = [sub_array for sublist in matxConf for sub_array in sublist]

        return flat_array

    def imageOrderByConf_Single(
        self, totalimage, arrimgname, arrdescriptors, arrkeypoints
    ):
        end = 0
        matxConf = []
        for i in range(totalimage):
            arrconfidence = []
            for j in range(totalimage):
                start = time.time()
                """
                # nmatches a.k.a number of matches, the higher the value will make the stitching and pairing more accurate
                # also note that the bigger the number of nmatches the longer the time will be to be calculated.
                # reccomended is 2000 matches for most images.
                """

                if arrimgname[i] == arrimgname[j]:
                    confidence = 0
                    arrconfidence.append(int(confidence))
                    continue

                # KNN
                if self.method == "knn":
                    try:
                        matches = self.dmc.BFMatchKNN(
                            arrdescriptors[i], j, arrdescriptors
                        )
                        confidence = self.dmc.knnConfidenceMatch(
                            matches, arrkeypoints[i], j, arrkeypoints
                        )
                    except:
                        confidence = 0

                # Normal
                elif self.method == "bf":
                    matches = self.dmc.BFMatch(
                        arrdescriptors[i], j, arrdescriptors, nmatches=2000
                    )
                    confidence = self.dmc.findConfidenceMatch(
                        matches, arrkeypoints[i], j, arrkeypoints
                    )

                arrconfidence.append(int(confidence * 1000))
                end += time.time() - start

            print(f"({i+1}/{totalimage})")

            matxConf.append(arrconfidence)
        avgtime = end / (totalimage * (totalimage - 1))
        print(f"avg time per process = {avgtime}")
        return matxConf
