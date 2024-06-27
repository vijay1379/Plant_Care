import flet as ft
import httpx
import aiofiles

async def main(page: ft.Page):
    page.title = "Leaf Image Classifier"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.window.width = 360
    page.window.height = 640
    page.bgcolor = "#F1F8E9"  # Light leaf green background

    selected_file = None

    def on_upload_result(e: ft.FilePickerResultEvent):
        nonlocal selected_file
        if e.files:
            selected_file = e.files[0]
            selected_files.value = selected_file.name
            display_image(selected_file)
            upload_button.disabled = False
            page.update()

    def display_image(file):
        image_view.src = file.path
        image_container.visible = True
        page.update()

    def reset_ui(e):
        nonlocal selected_file
        selected_file = None
        selected_files.value = ""
        image_view.src = None
        image_container.visible = False
        upload_button.disabled = True
        results_container.visible = False
        page.update()

    pick_files_dialog = ft.FilePicker(on_result=on_upload_result)
    selected_files = ft.Text(visible=True, size=14, color="#33691E")  # Dark green text

    async def send_image(e):
        nonlocal selected_file
        if not selected_file:
            return

        url = "https://1ca6ad14-a7c1-463d-9d41-b139318e32c1-00-146zgnp505wyg.pike.replit.dev/predict"

        try:
            async with httpx.AsyncClient() as client:
                async with aiofiles.open(selected_file.path, mode='rb') as file:
                    file_contents = await file.read()
                files = {"file": (selected_file.name, file_contents, "image/jpeg")}
                response = await client.post(url, files=files)

            if response.status_code == 200:
                json_response = response.json()
                predicted_class = json_response.get('predicted_class', 'Not found')
                confidence = json_response.get('confidence', 'N/A')
                
                # Format confidence only if it's a number
                if isinstance(confidence, (int, float)):
                    confidence_str = f"{confidence:.2f}"
                else:
                    confidence_str = str(confidence)
                
                response_text.value = f"Class: {predicted_class}\nConfidence: {confidence_str}"
            else:
                response_text.value = f"Error: {response.status_code} - {response.text}"
        except Exception as ex:
            response_text.value = f"An error occurred: {str(ex)}"

        results_container.visible = True
        page.update()

    upload_button = ft.ElevatedButton(
        "Classify",
        icon=ft.icons.CATEGORY,
        on_click=send_image,
        disabled=True,
        style=ft.ButtonStyle(
            color=ft.colors.WHITE,
            bgcolor="#4CAF50",  # Green button
        )
    )

    reset_button = ft.ElevatedButton(
        "Reset",
        icon=ft.icons.REFRESH,
        on_click=reset_ui,
        style=ft.ButtonStyle(
            color=ft.colors.WHITE,
            bgcolor="#FF7043",  # Orange button for contrast
        )
    )

    response_text = ft.Text(
        size=20,
        color="#1B5E20",  # Dark green text
        weight=ft.FontWeight.BOLD,
    )

    image_view = ft.Image(
        width=300,
        height=300,
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10),
    )

    image_container = ft.Container(
        content=image_view,
        visible=False,
        alignment=ft.alignment.center,
        bgcolor="#E8F5E9",  # Very light green background
        border_radius=10,
        padding=10,
    )

    results_container = ft.Container(
        content=ft.Column([
            ft.Text("Classification Result:", size=18, weight=ft.FontWeight.BOLD, color="#33691E"),
            ft.Container(
                content=response_text,
                bgcolor="#C8E6C9",  # Light green background
                border_radius=10,
                padding=15,
                alignment=ft.alignment.center,
            )
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
        visible=False,
        bgcolor="#E8F5E9",  # Very light green background
        border_radius=10,
        padding=20,
        margin=ft.margin.only(top=20),
    )

    page.overlay.append(pick_files_dialog)

    content = ft.Column(
        [
            ft.Container(
                content=ft.Row(
                    [ft.Icon(ft.icons.ECO, color="#33691E"), 
                     ft.Text("Leaf Classifier", size=24, weight=ft.FontWeight.BOLD, color="#33691E")],
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                padding=10,
                bgcolor="#C8E6C9"  # Light green header
            ),
            ft.Container(
                content=ft.Column(
                    [
                        ft.ElevatedButton(
                            "Choose Image",
                            icon=ft.icons.IMAGE_SEARCH,
                            on_click=lambda _: pick_files_dialog.pick_files(
                                allow_multiple=False,
                                allowed_extensions=["png", "jpg", "jpeg"],
                            ),
                            style=ft.ButtonStyle(
                                color=ft.colors.WHITE,
                                bgcolor="#8BC34A",  # Light green button
                            )
                        ),
                        selected_files,
                        image_container,
                        ft.Row(
                            [upload_button, reset_button],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                        ),
                        results_container,
                    ],
                    spacing=15,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=20,
            )
        ],
        spacing=0,
    )

    page.add(
        ft.ListView(
            [content, ft.Container(height=20)],  # Add some bottom padding
            expand=1,
            spacing=0,
            padding=0,
        )
    )

ft.app(target=main)