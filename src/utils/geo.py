# Coordinates for key locations
# Can be extended or loaded from CSV
LOCATIONS = {
    'Goyang_Baekseok': (126.787, 37.643),
    'Bundang_Jeongja': (127.110, 37.365),
    'Seoul_Nowon_Jungrang': (127.075, 37.625),
    # Add others as needed
}

def get_location_coords(name):
    return LOCATIONS.get(name)
