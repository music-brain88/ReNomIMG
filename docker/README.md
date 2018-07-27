# ReNomIMG Docker

This docker images allows you to setup and use
ReNomIMG GUI tool easily.

### Build Docker image

```shell
sh build.sh
```

### Run Docker image

```shell
sh run.sh [-d path_to_data_src_dir -s path_to_storage_dir -p port]
```

### Example:
```shell
sh run.sh [-d path_to_data_src_dir -s path_to_storage_dir -p port]
```

You can pass following arguments.
- `-d` : Path to the data source directory. This directory contains image files and label files.
- `-s` : Path to the data storage directory. Sqlite DB, trained weight and pretrained weight will be arranged into this directory.
- `-p` : The port number.

If no arguments are passed, directories named `datasrc` and `storage` will be created in
current directory, and the application uses `8080` port.
