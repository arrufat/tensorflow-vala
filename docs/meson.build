valadoc = find_program('valadoc', required: false)

if valadoc.found()
    valadoc_command = [valadoc, '--package-name', 'tensorflow',
                       '--package-version', meson.project_version(), '--pkg=glib-2.0',
                       '--pkg=gobject-2.0', '--pkg=tensorflow',
                       '--force', '--directory', '@OUTPUT@', '@INPUT@']
    custom_target('Devhelp book', command: valadoc_command + ['--doclet=devhelp'],
                  input: vapi_files,
                  output: 'devhelp')
    install_subdir(meson.current_build_dir() + '/devhelp/tensorflow', install_dir: 'share/devhelp/books')
endif
