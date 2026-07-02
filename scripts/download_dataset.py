from kneeassist.data.downloader import MRNetDownloader


def main():

    downloader = MRNetDownloader()

    downloader.download_sample()


if __name__ == "__main__":
    main()