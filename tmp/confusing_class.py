import random
import json
relation_classes = {
    "per:charges": [
        "per:sentence",
        "per:allegations",
        "per:indictment",
        "per:offense",
        "no_relation",
        "org:top_members/employees"
    ],
    "per:date_of_death": [
        "per:age",
        "per:country_of_birth",
        "per:origin",
        "no_relation",
        "per:city_of_death",
        "per:date_of_birth"
    ],
    "per:country_of_death": [
        "per:city_of_death",
        "per:stateorprovince_of_death",
        "per:origin",
        "no_relation",
        "per:stateorprovince_of_birth",
        "per:stateorprovinces_of_residence"
    ],
    "per:cause_of_death": [
        "per:age",
        "no_relation",
        "per:date_of_birth",
        "per:city_of_death",
        "per:stateorprovince_of_death",
        "per:country_of_death"
    ],
    "org:founded_by": [
        "org:shareholders",
        "per:other_family",
        "org:members",
        "org:political/religious_affiliation",
        "org:number_of_employees/members",
        "no_relation"
    ],
    "org:founded": [
        "org:city_of_branch",
        "org:stateorprovince_of_branch",
        "org:country_of_branch",
        "org:number_of_employees/members",
        "no_relation",
        "org:members"
    ],
    "per:city_of_death": [
        "no_relation",
        "per:country_of_death",
        "per:city_of_birth",
        "per:date_of_birth",
        "per:date_of_death",
        "per:origin"
    ],
    "per:stateorprovince_of_death": [
        "per:city_of_death",
        "per:country_of_death",
        "per:date_of_death",
        "per:city_of_birth",
        "per:stateorprovince_of_birth",
        "no_relation"
    ],
    "per:date_of_birth": [
        "per:age",
        "per:country_of_birth",
        "per:stateorprovince_of_birth",
        "no_relation",
        "per:date_of_death",
        "per:origin"
    ],
    "per:stateorprovince_of_birth": [
        "no_relation",
        "per:stateorprovinces_of_residence",
        "per:origin",
        "per:date_of_birth",
        "per:country_of_birth",
        "per:date_of_death"
    ],
    "per:country_of_birth": [
        "per:stateorprovince_of_birth",
        "no_relation",
        "per:date_of_birth",
        "per:stateorprovinces_of_residence",
        "per:origin",
        "per:city_of_death"
    ],
    "per:city_of_birth": [
        "per:stateorprovince_of_birth",
        "no_relation",
        "per:country_of_birth",
        "per:stateorprovinces_of_residence",
        "per:origin",
        "per:city_of_death"
    ],
    "org:shareholders": [
        "org:members",
        "org:founded_by",
        "org:political/religious_affiliation",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "no_relation"
    ],
    "per:other_family": [
        "per:siblings",
        "no_relation",
        "per:parents",
        "per:children",
        "per:employee_of",
        "per:city_of_birth"
    ],
    "per:title": [
        "org:top_members/employees",
        "org:political/religious_affiliation",
        "no_relation",
        "per:charges",
        "org:shareholders",
        "per:other_family"
    ],
    "org:dissolved": [
        "org:city_of_branch",
        "org:stateorprovince_of_branch",
        "org:country_of_branch",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "no_relation"
    ],
    "per:countries_of_residence": [
        "per:stateorprovinces_of_residence",
        "per:cities_of_residence",
        "per:origin",
        "per:country_of_birth",
        "per:country_of_death",
        "no_relation"
    ],
    "per:stateorprovinces_of_residence": [
        "no_relation",
        "per:cities_of_residence",
        "per:origin",
        "per:stateorprovince_of_birth",
        "per:stateorprovince_of_death",
        "org:founded"
    ],
    "per:cities_of_residence": [
        "no_relation",
        "per:stateorprovinces_of_residence",
        "per:origin",
        "per:city_of_birth",
        "per:city_of_death",
        "org:founded"
    ],
    "org:member_of": [
        "no_relation",
        "org:shareholders",
        "org:top_members/employees",
        "org:members",
        "org:number_of_employees/members",
        "org:founded_by"
    ],
    "per:religion": [
        "org:political/religious_affiliation",
        "per:origin",
        "per:other_family",
        "org:member_of",
        "org:shareholders",
        "per:title"
    ],
    "org:political/religious_affiliation": [
        "per:religion",
        "org:member_of",
        "org:shareholders",
        "per:title",
        "per:employee_of",
        "no_relation"
    ],
    "org:top_members/employees": [
        "per:title",
        "per:employee_of",
        "per:other_family",
        "no_relation",
        "org:shareholders",
        "org:political/religious_affiliation"
    ],
    "org:number_of_employees/members": [
        "org:shareholders",
        "org:members",
        "org:political/religious_affiliation",
        "org:founded_by",
        "org:top_members/employees",
        "no_relation"
    ],
    "per:schools_attended": [
        "org:member_of",
        "per:employee_of",
        "per:charges",
        "no_relation",
        "per:city_of_birth",
        "per:origin"
    ],
    "per:employee_of": [
        "per:other_family",
        "per:schools_attended",
        "per:title",
        "org:top_members/employees",
        "org:political/religious_affiliation",
        "no_relation"
    ],
    "per:siblings": [
        "per:other_family",
        "per:spouse",
        "per:parents",
        "per:children",
        "per:employee_of",
        "no_relation"
    ],
    "per:spouse": [
        "no_relation",
        "per:siblings",
        "per:parents",
        "per:children",
        "per:employee_of",
        "per:city_of_birth"
    ],
    "per:parents": [
        "per:other_family",
        "per:siblings",
        "per:spouse",
        "per:children",
        "per:employee_of",
        "no_relation"
    ],
    "per:children": [
        "per:other_family",
        "per:siblings",
        "no_relation",
        "per:parents",
        "per:employee_of",
        "per:city_of_birth"
    ],
    "org:alternate_names": [
        "org:members",
        "org:shareholders",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "org:dissolved",
        "org:founded_by"
    ],
    "org:members": [
        "org:shareholders",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "no_relation",
        "org:political/religious_affiliation",
        "org:founded_by"
    ],
    "per:origin": [
        "per:religion",
        "no_relation",
        "org:member_of",
        "org:shareholders",
        "org:top_members/employees",
        "org:number_of_employees/members"
    ],
    "org:website": [
        "org:member_of",
        "org:shareholders",
        "org:top_members/employees",
        "org:number_of_employees/members",
        "no_relation",
        "org:founded_by"
    ],
    "per:age": [
        "per:countries_of_residence",
        "per:stateorprovinces_of_residence",
        "per:cities_of_residence",
        "per:origin",
        "per:country_of_birth",
        "no_relation"
    ],
    "no_relation": [
        "org:political/religious_affiliation",
        "org:member_of",
        "org:shareholders",
        "per:title",
        "per:employee_of",
        "org:top_members/employees"
    ],
    "per:identity": [
        "no_relation",
        "per:spouse",
        "per:parents",
        "per:children",
        "per:employee_of",
        "per:city_of_birth"
    ],
    "org:stateorprovince_of_branch": [
        "org:city_of_branch",
        "org:country_of_branch",
        "org:number_of_employees/members",
        "no_relation",
        "org:members",
        "org:dissolved"
    ],
    "org:country_of_branch": [
        "org:city_of_branch",
        "org:stateorprovince_of_branch",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "org:members",
        "no_relation"
    ],
    "org:city_of_branch": [
        "org:stateorprovince_of_branch",
        "org:country_of_branch",
        "org:number_of_employees/members",
        "org:top_members/employees",
        "org:members",
        "org:dissolved"
    ]
}



def modify_relation_classes(relation_classes):
    modified_classes = relation_classes.copy()
    
    for key, entities in modified_classes.items():
        if key == "no_relation":
            continue
        
        if random.random() < 0.9:
            random_index = random.randint(0, len(entities) - 1)
            entities[random_index] = "no_relation"
    
    return modified_classes

# Test the function
modified_relation_classes = modify_relation_classes(relation_classes)

# Convert the modified relation classes to JSON string
json_str = json.dumps(modified_relation_classes, indent=4)

# Print the JSON string
print(json_str)