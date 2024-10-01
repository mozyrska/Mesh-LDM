addpath('./src');
unstructured_dir = './all_samples/vtk';
polydata_dir = './all_samples/vtk_polydata';

for case_num = 1:1000
    unstructured_file = fullfile(unstructured_dir, ['case_', num2str(case_num-1), '.vtk']);
    polydata_file = fullfile(polydata_dir, ['case_', num2str(case_num-1), '.vtk']);
    mesh = read_VTK(unstructured_file);
    write_VTK_POLYDATA(mesh, polydata_file);
end

