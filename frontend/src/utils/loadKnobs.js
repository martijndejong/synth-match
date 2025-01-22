import yaml from 'js-yaml';

export async function loadKnobsFromYaml(filePath) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`Failed to fetch YAML: ${response.statusText}`);
    }
    const yamlString = await response.text();
    const parsed = yaml.load(yamlString);
    return parsed.parameters || [];
  } catch (error) {
    console.error('Error loading knobs:', error);
    return [];
  }
}
