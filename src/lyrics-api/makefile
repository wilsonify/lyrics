.PHONY: clean try

clean:
	rm -rf build
	rm -rf dist

dist/__main__: clean
	pyinstaller --noconfirm --onedir --console __main__.spec --distpath dist --workpath build

try: dist/__main__
	cd dist/__main__ && ./__main__