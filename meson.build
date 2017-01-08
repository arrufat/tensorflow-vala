project('tensorflow', ['vala', 'c'],
        version: '0.12.1',
        default_options : ['buildtype=release'])

api_version = '0.12'

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')
tensorflow = dependency('tensorflow')

vapi_files = join_paths(meson.current_source_dir(), 'vapi', 'tensorflow.vapi')

add_global_arguments(['--vapidir=' + join_paths(meson.current_source_dir(), 'vapi')], language : 'vala')

# executable('tensorflow', 'examples/tensorflow.vala',
#     dependencies: [glib, gobject, tensorflow],
#     vala_args: ['--vapidir=' + join_paths(meson.current_source_dir(), 'vapi')])

subdir('tests')
subdir('docs')

install_subdir(join_paths(meson.current_source_dir(), 'vapi'), install_dir: 'share/vala')
install_subdir(join_paths(meson.current_source_dir(), 'pkgconfig'), install_dir: 'lib')
