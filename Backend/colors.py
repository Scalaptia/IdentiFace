from math import sqrt

skin_tone_categories = {
    "Claro": "#F6CEB0",
    "Justo": "#E9B591",
    "Medio": "#D29E7D",
    "Oliva": "#BB7952",
    "Oscuro": "#A45E2A"
}

seasons  = {
    "Invierno": {
        "Claro": ["#5F9EA0", "#FFB6C1", "#50C878", "#E5E4E2"],
        "Justo": ["#C0C0C0", "#0047AB", "#800020", "#FFFFFF"],
        "Medio": ["#36454F", "#800080", "#990000", "#0F0F0F"],
        "Oliva": ["#4169E1", "#FF69B4", "#708090", "#008080"],
        "Oscuro": ["#FFFFFF", "#000000", "#000080", "#9B111E"],
    },
    "Verano": {
        "Claro": ["#B0E0E6", "#E6E6FA", "#FFC0CB", "#D3D3D3"],
        "Justo": ["#FFFFE0", "#98FB98", "#FF69B4", "#87CEEB"],
        "Medio": ["#FFC0CB", "#E0B0FF", "#DDA0DD", "#2E8B57"],
        "Oliva": ["#191970", "#C8A2C8", "#98FF98", "#696969"],
        "Oscuro": ["#F5F5F5", "#F5F5DC", "#FFB6C1", "#C0C0C0"],
    },
    "Otoño": {
        "Claro": ["#C19A6B", "#556B2F", "#FF4500", "#FFF5EE"],
        "Justo": ["#FFDB58", "#B22222", "#228B22", "#F5F5DC"],
        "Medio": ["#D2B48C", "#B87333", "#008080", "#FF7F50"],
        "Oliva": ["#D2691E", "#FFBF00", "#708238", "#E97451"],
        "Oscuro": ["#8B4513", "#CD7F32", "#556B2F", "#8A3324"],
    },
    "Primavera": {
        "Claro": ["#FFE5B4", "#FF7F50", "#E0FFFF", "#FAEBD7"],
        "Justo": ["#FFD700", "#40E0D0", "#FF6347", "#FFFFF0"],
        "Medio": ["#FFDAB9", "#FA8072", "#00FFFF", "#98FB98"],
        "Oliva": ["#FF7F50", "#ADFF2F", "#FFE4B5", "#00FFFF"],
        "Oscuro": ["#FFD700", "#008000", "#00CED1", "#FF69B4"],
    }
}

class ColorFunctions:
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb_color):
        return '#{:02x}{:02x}{:02x}'.format(*rgb_color)
    
    @staticmethod
    def color_distance(color1, color2):
        return sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(color1, color2)))
    