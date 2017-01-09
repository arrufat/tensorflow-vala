valadoc = find_program('valadoc', required: false)

if valadoc.found()
    valadoc_command = [valadoc, '--package-name', 'tensorflow',
                       '--package-version', meson.project_version(), '--pkg=glib-2.0',
                       '--pkg=gobject-2.0', '--pkg=tensorflow',
                       '--force', '--directory', '@OUTPUT@', '@INPUT@']
    custom_target('API documentation', command: valadoc_command + ['--doclet=html'],
                  input: vapi_files,
                  output: 'api')
    # install_subdir(meson.current_build_dir() + '/api/', install_dir: 'share/doc/tensorflow-@0@'.format(api_version))
    custom_target('Devhelp book', command: valadoc_command + ['--doclet=devhelp',
        '&&', 'sed', '-i', meson.current_build_dir() + '/devhelp/tensorflow/tensorflow.devhelp2', '-e', 's/tensorflow/TensorFlow/',
        '&&', 'sed', '-i', meson.current_build_dir() + '/devhelp/tensorflow/index.htm', '-e', 's/<title>tensorflow/<Title>TensorFlow/',
        '&&', 'sed', '-i', meson.current_build_dir() + '/devhelp/tensorflow/index.htm', '-e', 's/tensorflow/TensorFlow/'],
                  input: vapi_files,
                  output: 'devhelp')
    # custom_target('Fix TensorFlow title case', command: ['sed', '-i', meson.current_build_dir() + '/devhelp/tensorflow/tensorflow.devhelp2', '-e', 's/tensorflow/TensorFlow/'],
    #               output: 'fix_devhelp')
    install_subdir(meson.current_build_dir() + '/devhelp/tensorflow', install_dir: 'share/devhelp/books')
endif
