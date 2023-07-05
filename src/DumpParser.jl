struct LammpsDump{HD,DD,FH}
    header_length::UInt32
    header_data::HD
    n_lines::UInt32
    n_samples::UInt32
    data_storage::DD
    file_handle::FH
end


# mutable struct MyTest
#     a
#     file_handle
#     function MyTest(a, path)
#     file_handle = open(path, "w")
#     finalizer(x -> close(x.file_handle), new(a, file_handle))
#     end
#     end

function LammpsDump(path; header_length = 9)
    header_data, n_lines = parse_dump_header(path, header_length)
    n_samples = UInt32(n_lines / (headear_data["N_atoms"] + header_length))

    # Build dataframe to hold data -- reuse memory
    dump_data = DataFrame(NamedTuple{Symbol.(header_data["fields"])}((zeros(header_data["N_atoms"]) 
        for i in range(1, length(header_data["fields"])))))

    file_handle = open(path, "r")

    return finalizer((ld) -> close(ld.file_handle), 
        LammpsDump(header_length, header_data, n_lines, n_samples, dump_data, file_handle))
end

Base.length(ld::LammpsDump) = ld.n_samples

function Base.iterate(ld::LammpsDump, state = 1)
    state > length(ld) ? nothing : (parse_timestep(ld, state), state + 1)
end


function parse_dump_header(dump_path, dump_header_len)
    file = open(dump_path, "r")
    n_lines = countlines(file) #need to close file after getting this data
    close(file)

    file = open(dump_path, "r")

    #parse first header for useful data
    header_data = Dict("N_atoms" => 0.0, "L_x" => [0.0,0.0], "L_y" => [0.0,0.0],
         "L_z" => [0.0,0.0], "fields" => nothing)
    for i in range(1, dump_header_len)
        line = readline(file)
        if i == 4
            header_data["N_atoms"] = parse(Int, line)
        elseif i == 6
            header_data["L_x"]  = [parse(Float64,x) for x in split(strip(line))]
        elseif i == 7
            header_data["L_y"] = [parse(Float64,y) for y in split(strip(line))]
        elseif i == 8
            header_data["L_z"] = [parse(Float64,z) for z in split(strip(line))]
        elseif i == 9
            header_data["fields"] = split(strip(line))[3:end]
        end
    end

    return header_data, n_lines
end

function parse_timestep!(ld::LammpsDump, sample_number)

    #Parse atom data
    for j in range(1, ld.header_data["N_atoms"])
        ld.data_storage[j,:] .= parse.(Float64, split(strip(readline(ld.file_handle))))
    end

    return ld.data_storage
end