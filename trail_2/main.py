import flet as ft
import requests
import os
import json
from PIL import Image

upload_dir = "uploaded_images"

def main(page: ft.Page):
    global upload_dir
    page.title = "Image Classifier"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10
    page.bgcolor = ft.colors.BLUE_GREY_900
    page.window.width = 360  # Typical width for a phone screen
    page.window.height = 640  # Typical height for a phone screen

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    def on_upload_result(e: ft.FilePickerResultEvent):
        if e.files:
            upload_files(e.files)

    def upload_files(files):
        upload_progress_text.visible = True
        upload_progress_text.value = "Uploading..."
        page.update()

        for file in files:
            file_path = os.path.join(upload_dir, file.name)
            selected_files.value = file.name
            display_image(file.path)

        upload_progress_text.value = "Upload completed!"
        upload_button.disabled = False
        reset_button.disabled = False
        page.update()

    def display_image(file_path):
        with Image.open(file_path) as img:
            img.thumbnail((300, 300))
            save_path = os.path.join(upload_dir, os.path.basename(file_path))
            img.save(save_path)
        
        image_view.src = save_path
        image_view.visible = True
        page.update()

    def reset_ui(e):
        selected_files.value = ""
        upload_progress_text.value = ""
        upload_progress_text.visible = False
        image_view.src = None
        image_view.visible = False
        upload_button.disabled = True
        reset_button.disabled = True
        response_text.value = ""
        page.update()

    pick_files_dialog = ft.FilePicker(on_result=on_upload_result)
    selected_files = ft.Text(visible=True, size=14)
    upload_progress_text = ft.Text(visible=False, size=14)

    def send_image(e):
        if not selected_files.value:
            response_text.value = "Please select an image first."
            response_text.update()
            return

        file_path = os.path.join(upload_dir, selected_files.value)
        url = "https://1ca6ad14-a7c1-463d-9d41-b139318e32c1-00-146zgnp505wyg.pike.replit.dev/predict"

        try:
            with open(file_path, "rb") as file:
                files = {"file": (os.path.basename(file_path), file, "image/jpeg")}
                response = requests.post(url, files=files)

            if response.status_code == 200:
                try:
                    json_response = json.loads(response.text)
                    predicted_class = json_response.get('predicted_class', 'Not found')
                    response_text.value = f"Predicted Class: {predicted_class}"
                except json.JSONDecodeError:
                    response_text.value = "Error: Unable to parse JSON response"
            else:
                response_text.value = f"Error: {response.status_code} - {response.text}"
        except Exception as ex:
            response_text.value = f"An error occurred: {str(ex)}"

        response_text.update()

    upload_button = ft.ElevatedButton(
        "Classify",
        icon=ft.icons.UPLOAD_FILE,
        on_click=send_image,
        disabled=True,
        style=ft.ButtonStyle(
            color=ft.colors.WHITE,
            bgcolor=ft.colors.BLUE_600,
        )
    )

    reset_button = ft.ElevatedButton(
        "Reset",
        icon=ft.icons.REFRESH,
        on_click=reset_ui,
        disabled=True,
        style=ft.ButtonStyle(
            color=ft.colors.WHITE,
            bgcolor=ft.colors.RED_400,
        )
    )

    response_text = ft.Text(
        size=16,
        color=ft.colors.WHITE,
        weight=ft.FontWeight.BOLD,
    )

    image_view = ft.Image(
        visible=False,
        width=300,
        height=300,
        fit=ft.ImageFit.CONTAIN,
    )

    page.overlay.append(pick_files_dialog)

    page.add(
        ft.Column(
            [
                ft.Text("Image Classifier", size=24, weight=ft.FontWeight.BOLD),
                ft.ElevatedButton(
                    "Choose Image",
                    icon=ft.icons.IMAGE_SEARCH,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False,
                        allowed_extensions=["png", "jpg", "jpeg"],
                    ),
                    style=ft.ButtonStyle(
                        color=ft.colors.WHITE,
                        bgcolor=ft.colors.GREEN_600,
                    )
                ),
                selected_files,
                upload_progress_text,
                image_view,
                ft.Row(
                    [upload_button, reset_button],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                ft.Container(height=10),
                ft.Text("Classification Result:", size=18, weight=ft.FontWeight.BOLD),
                response_text,
            ],
            spacing=10,
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )
    )

ft.app(target=main)