from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth
from dotenv import load_dotenv
import os

from nexa_py_sentimotion_mapper.sentimotion_mapper import Mapper


if __package__:
    from .config import video_id_pattern
else:
    from config import video_id_pattern


def get_all_items_in_folder(client, folder_id):
    all_items = []
    offset = 0
    limit = 100

    while True:
        response = client.folders.get_folder_items(folder_id=folder_id, offset=offset, limit=limit)
        all_items.extend(response.entries)

        if len(response.entries) < limit:
            break

        offset += limit

    return all_items


# def iterate_files(client):
#     for video_id_folder in iterate_video_ids(client):  # Unroll the generator here
#         for video_id_content in get_all_items_in_folder(client, video_id_folder.id):
#             if video_id_content.name == "normalized audio_video" or video_id_content.name == "normalized audio-video clips":
#                 for video_file in get_all_items_in_folder(client, video_id_content.id):
#                     yield video_file


def iterate_edited_video_ids(client):
    for item in get_all_items_in_folder(client, "0"):
        if item.name == "Edited":
            for folder in get_all_items_in_folder(client, item.id):
                if video_id_pattern.match(folder.name):
                    yield folder.name


def iterate_validation_video_ids(client):
    for item in get_all_items_in_folder(client, "0"):
        if item.name == "Edited":
            for subitem in get_all_items_in_folder(client, item.id):
                if subitem.name == "for validation study":
                    for folder in get_all_items_in_folder(client, subitem.id):
                        if video_id_pattern.match(folder.name):
                            yield folder.name


def get_box_video_ids():
    load_dotenv()

    DEV_TOKEN = os.getenv("BOX_DEV_TOKEN")
    if not DEV_TOKEN:
        raise RuntimeError("BOX_DEV_TOKEN is required to list Box video IDs.")
    auth: BoxDeveloperTokenAuth = BoxDeveloperTokenAuth(token=DEV_TOKEN)
    client: BoxClient = BoxClient(auth=auth)

    edited_ids = set(iterate_edited_video_ids(client))
    validation_ids = set(iterate_validation_video_ids(client))

    return edited_ids, validation_ids


def main():
    edited_ids, validation_ids = get_box_video_ids()

    print(f"Edited: {len(edited_ids)}")
    print(f"Validation: {len(validation_ids)}")

    # IDs present in Edited but not in the validation folder
    print("Only in Edited:")
    print(sorted(edited_ids - validation_ids))

    # IDs present in validation but not in Edited
    print("Only in Validation:")
    print(sorted(validation_ids - edited_ids))


if __name__ == "__main__":
    main()
