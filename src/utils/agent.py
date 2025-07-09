from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from typing import Dict, Any, Optional


def render_template(template_name: str, template_args: Optional[Dict[str, Any]] = None) -> str:
    """
    Render a Jinja2 template from the templates directory.
    
    Args:
        template_name (str): Name of the template file (with or without .j2 extension)
        template_args (dict, optional): Dictionary of arguments to pass to the template
        
    Returns:
        str: Rendered template text
        
    Raises:
        TemplateNotFound: If the template file doesn't exist
        Exception: If there's an error rendering the template
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    templates_dir = project_root / "templates"
    
    if not template_name.endswith('.j2'):
        template_name += '.j2'
    
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        trim_blocks=True,
        lstrip_blocks=True
    )
    
    try:
        template = env.get_template(template_name)
        rendered_text = template.render(**(template_args or {}))
        return rendered_text.strip()
        
    except TemplateNotFound:
        raise TemplateNotFound(f"Template '{template_name}' not found in {templates_dir}")
    except Exception as e:
        raise Exception(f"Error rendering template '{template_name}': {str(e)}")


def get_available_templates() -> list[str]:
    """
    Get a list of available template files in the templates directory.
    
    Returns:
        list[str]: List of template filenames (without .j2 extension)
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    templates_dir = project_root / "templates"
    
    if not templates_dir.exists():
        return []
    
    templates = []
    for file_path in templates_dir.glob("*.j2"):
        templates.append(file_path.stem)
    
    return sorted(templates)
