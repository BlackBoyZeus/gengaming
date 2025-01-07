// Theme configuration for GameGen-X web interface
// Version: 1.0.0

// Global constants
const SPACING_UNIT = 8;
const BASE_FONT_SIZE = 16;
const TRANSITION_BASE = '0.2s';

// Color palette interfaces
interface ThemeColors {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  hover: string;
  active: string;
  focus: string;
  error: string;
  success: string;
  warning: string;
  overlay: string;
  accent: string;
}

// Typography system interface
interface Typography {
  fontFamily: string;
  fontFamilyMono: string;
  baseSize: string;
  sizeScale: {
    xs: string;
    sm: string;
    base: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
  };
  lineHeight: {
    none: number;
    tight: number;
    normal: number;
    loose: number;
  };
  fontWeight: {
    normal: number;
    medium: number;
    bold: number;
  };
  letterSpacing: {
    tight: string;
    normal: string;
    wide: string;
  };
}

// Spacing system interface
interface Spacing {
  unit: number;
  scale: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
  };
  inset: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
  };
  layout: {
    container: string;
    content: string;
    wide: string;
  };
}

// Visual effects interface
interface Effects {
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    full: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
    inner: string;
  };
  transitions: {
    fast: string;
    normal: string;
    slow: string;
  };
  animations: {
    fade: string;
    slide: string;
    scale: string;
  };
  glows: {
    primary: string;
    success: string;
    error: string;
  };
}

// Complete theme interface
export interface Theme {
  colors: ThemeColors;
  typography: Typography;
  spacing: Spacing;
  effects: Effects;
  name: string;
  version: string;
}

// Light theme configuration
export const lightTheme: Theme = {
  name: 'light',
  version: '1.0.0',
  colors: {
    primary: '#4A90E2',
    secondary: '#50E3C2',
    background: '#F5F7FA',
    surface: '#FFFFFF',
    text: '#2C3E50',
    textSecondary: '#7F8C8D',
    border: '#E2E8F0',
    hover: '#3498DB',
    active: '#2980B9',
    focus: 'rgba(74, 144, 226, 0.4)',
    error: '#E74C3C',
    success: '#2ECC71',
    warning: '#F1C40F',
    overlay: 'rgba(44, 62, 80, 0.8)',
    accent: '#9B59B6'
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    fontFamilyMono: 'SFMono-Regular, Menlo, Monaco, Consolas, monospace',
    baseSize: `${BASE_FONT_SIZE}px`,
    sizeScale: {
      xs: '0.75rem',
      sm: '0.875rem',
      base: '1rem',
      lg: '1.125rem',
      xl: '1.25rem',
      '2xl': '1.5rem',
      '3xl': '2rem'
    },
    lineHeight: {
      none: 1,
      tight: 1.25,
      normal: 1.5,
      loose: 2
    },
    fontWeight: {
      normal: 400,
      medium: 500,
      bold: 700
    },
    letterSpacing: {
      tight: '-0.025em',
      normal: '0',
      wide: '0.025em'
    }
  },
  spacing: {
    unit: SPACING_UNIT,
    scale: {
      xs: `${SPACING_UNIT / 2}px`,
      sm: `${SPACING_UNIT}px`,
      md: `${SPACING_UNIT * 2}px`,
      lg: `${SPACING_UNIT * 3}px`,
      xl: `${SPACING_UNIT * 4}px`,
      '2xl': `${SPACING_UNIT * 6}px`,
      '3xl': `${SPACING_UNIT * 8}px`
    },
    inset: {
      xs: `${SPACING_UNIT / 2}px`,
      sm: `${SPACING_UNIT}px`,
      md: `${SPACING_UNIT * 2}px`,
      lg: `${SPACING_UNIT * 3}px`
    },
    layout: {
      container: '1280px',
      content: '768px',
      wide: '1440px'
    }
  },
  effects: {
    borderRadius: {
      sm: '4px',
      md: '6px',
      lg: '8px',
      full: '9999px'
    },
    shadows: {
      sm: '0 1px 3px rgba(0, 0, 0, 0.1)',
      md: '0 4px 6px rgba(0, 0, 0, 0.1)',
      lg: '0 10px 15px rgba(0, 0, 0, 0.1)',
      inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.1)'
    },
    transitions: {
      fast: `${TRANSITION_BASE} ease-in-out`,
      normal: `${TRANSITION_BASE} * 1.5 ease-in-out`,
      slow: `${TRANSITION_BASE} * 2 ease-in-out`
    },
    animations: {
      fade: 'fade 0.3s ease-in-out',
      slide: 'slide 0.3s ease-in-out',
      scale: 'scale 0.3s ease-in-out'
    },
    glows: {
      primary: '0 0 15px rgba(74, 144, 226, 0.6)',
      success: '0 0 15px rgba(46, 204, 113, 0.6)',
      error: '0 0 15px rgba(231, 76, 60, 0.6)'
    }
  }
};

// Dark theme configuration
export const darkTheme: Theme = {
  name: 'dark',
  version: '1.0.0',
  colors: {
    primary: '#5B9FE2',
    secondary: '#50E3C2',
    background: '#1A1F2C',
    surface: '#2C3E50',
    text: '#ECF0F1',
    textSecondary: '#BDC3C7',
    border: '#34495E',
    hover: '#3498DB',
    active: '#2980B9',
    focus: 'rgba(91, 159, 226, 0.4)',
    error: '#E74C3C',
    success: '#2ECC71',
    warning: '#F1C40F',
    overlay: 'rgba(26, 31, 44, 0.9)',
    accent: '#9B59B6'
  },
  typography: lightTheme.typography,
  spacing: lightTheme.spacing,
  effects: {
    ...lightTheme.effects,
    shadows: {
      sm: '0 1px 3px rgba(0, 0, 0, 0.3)',
      md: '0 4px 6px rgba(0, 0, 0, 0.3)',
      lg: '0 10px 15px rgba(0, 0, 0, 0.3)',
      inner: 'inset 0 2px 4px rgba(0, 0, 0, 0.3)'
    },
    glows: {
      primary: '0 0 20px rgba(91, 159, 226, 0.4)',
      success: '0 0 20px rgba(46, 204, 113, 0.4)',
      error: '0 0 20px rgba(231, 76, 60, 0.4)'
    }
  }
};